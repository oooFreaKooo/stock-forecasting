import type { ChartPoint, ChartSeriesResponse } from '~/types/radar'
import { roundPrice } from '~/utils/format'

function parseUtcMs(iso: string): number {
  return Date.parse(iso.endsWith('Z') ? iso : `${iso}Z`)
}

function hourBucketUtc(iso: string): number {
  const d = new Date(parseUtcMs(iso))
  return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate(), d.getUTCHours())
}

function bucketToIso(bucketMs: number): string {
  return new Date(bucketMs).toISOString()
}

/** Last 5m close in each UTC hour (matches API resample_chart_points_to_1h). */
export function resamplePointsTo1h(points: ChartPoint[]): ChartPoint[] {
  if (!points.length) return []
  const sorted = [...points].sort((a, b) => parseUtcMs(a.date) - parseUtcMs(b.date))
  const byHour = new Map<number, ChartPoint>()
  for (const pt of sorted) {
    byHour.set(hourBucketUtc(pt.date), pt)
  }
  return [...byHour.entries()]
    .sort(([a], [b]) => a - b)
    .map(([bucketMs, pt]) => ({
      date: bucketToIso(bucketMs),
      close: roundPrice(pt.close) ?? pt.close,
    }))
}

/** Legacy fallback when bundle lacks intraday_1h — resamples the single API model line. */
export function resampleIntradayChartTo1h(chart5m: ChartSeriesResponse): ChartSeriesResponse {
  const points = resamplePointsTo1h(chart5m.points)
  const forecast = chart5m.forecast
  const valPoints = resamplePointsTo1h(chart5m.validation?.points ?? [])
  const fwdPoints = resamplePointsTo1h(forecast.points)
  const modelPoints = resamplePointsTo1h(chart5m.model?.points ?? [])

  return {
    ...chart5m,
    interval: '1h',
    points,
    model: {
      engine: chart5m.model?.engine ?? forecast.engine,
      points: modelPoints,
      backtest_bars: Math.max(0, modelPoints.length - fwdPoints.length - 1),
      forward_bars: fwdPoints.length,
    },
    forecast: {
      ...forecast,
      horizon_bars: fwdPoints.length,
      points: fwdPoints,
    },
    validation: chart5m.validation
      ? { ...chart5m.validation, points: valPoints }
      : undefined,
    meta: {
      ...chart5m.meta,
      rows: points.length,
      validation_bars: valPoints.length,
      forecast_bars: fwdPoints.length,
      note: '1H view resampled from canonical 5M AI forecast (no second model run).',
    },
  }
}
