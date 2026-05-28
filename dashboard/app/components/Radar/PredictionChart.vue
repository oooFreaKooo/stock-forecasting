<script setup lang="ts">
import type { ApexOptions } from 'apexcharts'
import type {
  ChartInterval,
  ChartPoint,
  ChartSeriesResponse,
  ChartValidationMetrics,
  Prediction,
} from '~/types/radar'
import {
  formatChartIntradayAxis,
  formatChartIntradayDetail,
  formatPrice,
  formatPriceAxis,
  roundPrice,
} from '~/utils/format'
import { resampleIntradayChartTo1h } from '~/utils/chartResample'

const props = defineProps<{
  prediction: Prediction
  reloadToken?: number
}>()

const api = useRadarApi()

type LinePoint = { x: number; y: number }

const interval = ref<ChartInterval>('5m')
const chartMeta = ref<ChartSeriesResponse['meta'] | null>(null)
const forecastEngine = ref<string | null>(null)
const loading = ref(false)
const loadError = ref<string | null>(null)

const pricePoints = shallowRef<LinePoint[]>([])
const modelPoints = shallowRef<LinePoint[]>([])
const validationMetrics = ref<ChartValidationMetrics | null>(null)
const tooltipDates = shallowRef<string[]>([])
const useCalendarAxis = ref(false)
const axisStepMs = shallowRef(300_000)
const forecastMarkerX = shallowRef<number | null>(null)
const chartRef = useTemplateRef<{ chart?: ChartRef }>('chart')

let fetchAbort: AbortController | null = null

type ChartBundleCache = {
  symbol: string
  intraday: ChartSeriesResponse
  intraday1h: ChartSeriesResponse
  daily: ChartSeriesResponse
}

const chartBundleCache = shallowRef<ChartBundleCache | null>(null)

const INTERVALS: { value: ChartInterval; label: string; stepMs: number; defaultBars: number }[] = [
  { value: '5m', label: '5M', stepMs: 300_000, defaultBars: 156 },
  { value: '1h', label: '1H', stepMs: 3_600_000, defaultBars: 120 },
  { value: '1d', label: '1D', stepMs: 86_400_000, defaultBars: 90 },
]

const STEP_BY_INTERVAL = Object.fromEntries(INTERVALS.map(i => [i.value, i.stepMs])) as Record<ChartInterval, number>
const DEFAULT_BARS = Object.fromEntries(INTERVALS.map(i => [i.value, i.defaultBars])) as Record<ChartInterval, number>

const barCount = computed(() => pricePoints.value.length)
const forwardBarCount = computed(() => {
  const marker = forecastMarkerX.value
  if (marker == null) return 0
  return modelPoints.value.filter(p => p.x > marker).length
})
const hasData = computed(() => pricePoints.value.length > 0)
const chartKey = computed(() => `${props.prediction.symbol}-${interval.value}-${props.reloadToken ?? 0}`)

function parseUtcMs(iso: string): number {
  return Date.parse(iso.endsWith('Z') ? iso : `${iso}Z`)
}

function labelForAxisX(x: number): string {
  return formatChartIntradayAxis(new Date(x).toISOString(), interval.value !== '5m')
}

function tooltipForAxisX(x: number): string {
  return formatChartIntradayDetail(new Date(x).toISOString())
}

function computeBounds(points: LinePoint[]): { min: number; max: number } | null {
  if (!points.length) return null
  const ys = points.map(p => p.y)
  const min = Math.min(...ys)
  const max = Math.max(...ys)
  const pad = (max - min) * 0.08 || max * 0.015
  return { min: min - pad, max: max + pad }
}

function boundsForXRange(minX: number, maxX: number): { min: number; max: number } | null {
  const inRange = (pts: LinePoint[]) => pts.filter(p => p.x >= minX && p.x <= maxX)
  return computeBounds([
    ...inRange(pricePoints.value),
    ...inRange(modelPoints.value),
  ])
}

type ChartRef = {
  zoomX: (min: number, max: number) => void
  updateOptions: (opts: ApexOptions, redraw?: boolean, animate?: boolean, updateSyncedCharts?: boolean) => void
}

function fitYAxis(chart: ChartRef, minX: number, maxX: number) {
  const bounds = boundsForXRange(minX, maxX)
  if (!bounds) return
  chart.updateOptions({ yaxis: { min: bounds.min, max: bounds.max } }, false, false, false)
}

function defaultXRange(prices: LinePoint[], stepMs: number, forecastBars: number): { min: number; max: number } {
  const window = Math.min(DEFAULT_BARS[interval.value], prices.length)
  const slice = prices.slice(Math.max(0, prices.length - window))
  const start = slice[0]?.x ?? prices[0]!.x
  const end = prices[prices.length - 1]!.x
  return {
    min: start - stepMs * 0.5,
    max: end + forecastBars * stepMs + stepMs * 0.5,
  }
}

function fitDefaultView(chart: ChartRef) {
  const priceBars = pricePoints.value.length
  if (!priceBars) return
  const { min, max } = defaultXRange(pricePoints.value, axisStepMs.value, forwardBarCount.value)
  chart.zoomX(min, max)
  fitYAxis(chart, min, max)
}

function buildChartData(
  history: ChartPoint[],
  model: ChartPoint[],
  iv: ChartInterval,
) {
  const stepMs = STEP_BY_INTERVAL[iv]
  axisStepMs.value = stepMs
  useCalendarAxis.value = true

  const prices: LinePoint[] = []
  for (const bar of history) {
    const close = roundPrice(bar.close)
    if (close == null) continue
    prices.push({ x: parseUtcMs(bar.date), y: close })
  }

  const modelLine: LinePoint[] = []
  for (const pt of model) {
    const close = roundPrice(pt.close)
    if (close == null) continue
    modelLine.push({ x: parseUtcMs(pt.date), y: close })
  }

  const lastHist = prices.at(-1)
  let markerX: number | null = null
  if (lastHist && modelLine.some(p => p.x > lastHist.x)) {
    markerX = lastHist.x
  }

  tooltipDates.value = []
  pricePoints.value = prices
  modelPoints.value = modelLine
  forecastMarkerX.value = markerX
}

const validationSummary = computed(() => {
  const m = validationMetrics.value
  if (!m || !m.n_points) return null
  const dir = m.direction_accuracy != null ? `${(m.direction_accuracy * 100).toFixed(1)}% dir` : null
  const mae = m.mae != null ? `MAE ${formatPrice(m.mae)}` : null
  return [mae, dir].filter(Boolean).join(' · ')
})

const chartSeries = computed<ApexOptions['series']>(() => {
  const series: ApexOptions['series'] = [
    { name: 'Actual', type: 'line', data: pricePoints.value },
  ]
  if (modelPoints.value.length) {
    series.push({
      name: `AI forecast (${forecastEngine.value ?? 'baseline'})`,
      type: 'line',
      data: modelPoints.value,
    })
  }
  return series
})

const seriesStroke = computed(() => {
  const n = chartSeries.value?.length ?? 1
  if (n === 2) return { width: [2, 2.5] as number[], dash: [0, 6] as number[] }
  return { width: [2] as number[], dash: [0] as number[] }
})

const chartOptions = computed<ApexOptions>(() => ({
  chart: {
    id: `radar-${props.prediction.symbol}-${interval.value}`,
    type: 'line',
    height: 560,
    fontFamily: 'inherit',
    parentHeightOffset: 0,
    redrawOnParentResize: true,
    animations: { enabled: barCount.value < 200, speed: 200 },
    toolbar: {
      show: true,
      autoSelected: 'zoom',
      tools: {
        download: true,
        selection: true,
        zoom: true,
        zoomin: true,
        zoomout: true,
        pan: true,
        reset: true,
      },
    },
    zoom: {
      enabled: true,
      type: 'x',
      autoScaleYaxis: true,
      allowMouseWheelZoom: true,
    },
    events: {
      zoomed(chart, opts) {
        const minX = opts?.xaxis?.min
        const maxX = opts?.xaxis?.max
        if (minX == null || maxX == null) return
        fitYAxis(chart as ChartRef, minX, maxX)
      },
      scrolled(chart, opts) {
        const minX = opts?.xaxis?.min
        const maxX = opts?.xaxis?.max
        if (minX == null || maxX == null) return
        fitYAxis(chart as ChartRef, minX, maxX)
      },
      beforeResetZoom(chart) {
        window.setTimeout(() => fitDefaultView(chart as ChartRef), 0)
      },
    },
  },
  stroke: {
    width: seriesStroke.value.width,
    curve: 'straight',
    dashArray: seriesStroke.value.dash,
  },
  colors: ['#2563eb', '#8b5cf6'],
  grid: {
    borderColor: 'var(--color-border)',
    strokeDashArray: 4,
    padding: { left: 8, right: 16 },
  },
  xaxis: {
    type: 'datetime',
    tickAmount: interval.value === '5m' ? 14 : 10,
    labels: {
      datetimeUTC: true,
      hideOverlappingLabels: true,
      rotate: -35,
      style: { fontSize: '11px' },
      formatter: (val: string) => labelForAxisX(Number(val)),
    },
    crosshairs: {
      show: true,
      stroke: { color: 'var(--color-border)', width: 1, dashArray: 4 },
    },
    tooltip: { enabled: false },
  },
  yaxis: {
    labels: {
      formatter: (v: number) => formatPriceAxis(v),
      style: { fontSize: '11px' },
    },
    tickAmount: 8,
    forceNiceScale: true,
    decimalsInFloat: 2,
    crosshairs: {
      show: true,
      stroke: { color: 'var(--color-border)', width: 1, dashArray: 4 },
    },
    tooltip: { enabled: false },
  },
  legend: {
    show: modelPoints.value.length > 0,
    position: 'top',
    horizontalAlign: 'left',
  },
  markers: {
    size: 0,
    hover: { size: 4 },
  },
  tooltip: {
    enabled: true,
    shared: true,
    intersect: false,
    followCursor: true,
    hideEmptySeries: true,
    fixed: { enabled: false },
    x: { show: true, formatter: (val: number) => tooltipForAxisX(val) },
    y: { formatter: (v: number) => formatPrice(v) },
    marker: { show: true },
  },
  annotations: forecastMarkerX.value != null
    ? {
        xaxis: [{
          x: forecastMarkerX.value,
          borderColor: '#94a3b8',
          strokeDashArray: 4,
          label: {
            text: 'Forecast →',
            style: { fontSize: '11px', fontWeight: 600, color: '#64748b', background: 'transparent' },
          },
        }],
      }
    : undefined,
}))

function chartForInterval(cache: ChartBundleCache, iv: ChartInterval): ChartSeriesResponse {
  if (iv === '1d') return cache.daily
  if (iv === '1h') return cache.intraday1h ?? resampleIntradayChartTo1h(cache.intraday)
  return cache.intraday
}

function applyChartResponse(res: ChartSeriesResponse, iv: ChartInterval) {
  buildChartData(res.points, res.model?.points ?? [], iv)
  chartMeta.value = res.meta
  forecastEngine.value = res.model?.engine ?? res.forecast?.engine ?? res.meta.forecast_engine ?? null
  validationMetrics.value = res.validation?.metrics ?? null
}

function applyIntervalView() {
  const cache = chartBundleCache.value
  if (!cache || cache.symbol !== props.prediction.symbol) return
  applyChartResponse(chartForInterval(cache, interval.value), interval.value)
}

async function loadChart() {
  fetchAbort?.abort()
  fetchAbort = new AbortController()

  loading.value = true
  loadError.value = null

  const symbol = props.prediction.symbol

  try {
    const bundle = await api.fetchChartBundle(symbol)
    if (fetchAbort.signal.aborted) return
    chartBundleCache.value = {
      symbol: bundle.symbol,
      intraday: bundle.intraday,
      intraday1h: bundle.intraday_1h,
      daily: bundle.daily,
    }
    applyIntervalView()
  } catch (e: unknown) {
    if (fetchAbort.signal.aborted) return
    const raw = e instanceof Error ? e.message : 'Failed to load chart data'
    loadError.value = raw.toLowerCase().includes('timeout') || raw.toLowerCase().includes('aborted')
      ? 'Chart request timed out. Retry shortly or check /tmp/radar-api.log.'
      : raw
    chartBundleCache.value = null
    pricePoints.value = []
    modelPoints.value = []
    validationMetrics.value = null
    tooltipDates.value = []
    chartMeta.value = null
    forecastEngine.value = null
  } finally {
    if (!fetchAbort.signal.aborted) {
      loading.value = false
    }
  }
}

watch(interval, () => {
  if (chartBundleCache.value?.symbol === props.prediction.symbol) {
    applyIntervalView()
    return
  }
  loadChart()
})
watch(() => props.prediction.symbol, () => {
  chartBundleCache.value = null
  loadChart()
}, { immediate: true })
watch(() => props.reloadToken, () => {
  chartBundleCache.value = null
  loadChart()
})

watch(
  () => [pricePoints.value.length, modelPoints.value.length],
  async () => {
    if (!pricePoints.value.length) return
    await nextTick()
    const chart = chartRef.value?.chart
    if (chart) fitDefaultView(chart)
  },
)

onBeforeUnmount(() => {
  fetchAbort?.abort()
})
</script>

<template>
  <div class="flex flex-col gap-4">
    <div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
      <div class="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
        <div>
          <span class="text-muted-foreground">Last close</span>
          <span class="ml-2 font-semibold tabular-nums">{{ formatPrice(prediction.last_close) }}</span>
        </div>
        <div>
          <span class="text-muted-foreground">Signal 1d</span>
          <span
            class="ml-2 font-semibold tabular-nums"
            :class="(prediction.forecast_return_1d ?? 0) >= 0 ? 'text-emerald-600' : 'text-red-500'"
          >
            {{ prediction.forecast_return_1d != null ? `${(prediction.forecast_return_1d * 100).toFixed(2)}%` : '—' }}
          </span>
        </div>
        <div v-if="validationSummary" class="text-violet-600 dark:text-violet-400">
          Backtest: {{ validationSummary }}
        </div>
        <div v-if="chartMeta" class="text-muted-foreground">
          {{ chartMeta.rows }} bars · {{ chartMeta.validation_bars ?? 0 }} backtest · {{ chartMeta.forecast_bars ?? 0 }} forward
        </div>
      </div>

      <UiTabs v-model="interval" class="w-full sm:w-auto">
        <UiTabsList class="grid w-full grid-cols-3 sm:w-auto">
          <UiTabsTrigger v-for="item in INTERVALS" :key="item.value" :value="item.value">
            {{ item.label }}
          </UiTabsTrigger>
        </UiTabsList>
      </UiTabs>
    </div>

    <p class="text-xs text-muted-foreground">
      Blue = actual price · Violet dashed = one AI path (walk-forward backtest, then live forward from the last close).
      <template v-if="interval !== '1d'">
        Timestamps are UTC (Europe/Berlin in tooltips).
      </template>
      <span v-if="chartMeta?.note"> — {{ chartMeta.note }}</span>
    </p>

    <UiCard v-if="loadError" class="border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-950/30">
      <UiCardContent class="py-3 text-sm text-red-700 dark:text-red-300">
        {{ loadError }}
      </UiCardContent>
    </UiCard>

    <div class="radar-chart relative w-full overflow-visible">
      <div
        v-if="loading"
        class="absolute inset-0 z-10 flex h-[560px] items-center justify-center rounded-lg bg-background/70 backdrop-blur-sm"
      >
        <span class="text-sm text-muted-foreground">Loading chart (5M AI + daily)…</span>
      </div>

      <ClientOnly>
        <UiApexchart
          v-if="hasData || loading"
          ref="chart"
          :key="chartKey"
          type="line"
          :height="560"
          width="100%"
          :options="chartOptions"
          :series="chartSeries"
        />
        <div
          v-else-if="!loadError"
          class="flex h-[560px] items-center justify-center text-sm text-muted-foreground"
        >
          No chart data for this interval.
        </div>
        <template #fallback>
          <div class="flex h-[560px] items-center justify-center text-sm text-muted-foreground">
            Loading chart…
          </div>
        </template>
      </ClientOnly>
    </div>
  </div>
</template>

<style scoped>
.radar-chart :deep(.apexcharts-canvas),
.radar-chart :deep(.vue-apexcharts) {
  width: 100%;
  overflow: visible !important;
}

.radar-chart :deep(.apexcharts-toolbar) {
  z-index: 5;
}

.radar-chart :deep(.apexcharts-tooltip) {
  font-variant-numeric: tabular-nums;
  pointer-events: none;
}

.radar-chart :deep(.apexcharts-yaxistooltip) {
  display: none !important;
}
</style>
