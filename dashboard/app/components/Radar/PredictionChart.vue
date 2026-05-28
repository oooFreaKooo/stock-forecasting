<script setup lang="ts">
import type { ApexOptions } from 'apexcharts'
import type { ChartInterval, ChartPoint, ChartSeriesResponse, Prediction } from '~/types/radar'
import {
  formatChartIntradayAxis,
  formatChartIntradayDetail,
  formatPrice,
  formatPriceAxis,
  roundPrice,
} from '~/utils/format'

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
const forecastPoints = shallowRef<LinePoint[]>([])
const tooltipDates = shallowRef<string[]>([])
const axisStepMs = shallowRef(300_000)
const forecastMarkerX = shallowRef<number | null>(null)
const chartRef = useTemplateRef<{ chart?: ChartRef }>('chart')

let fetchAbort: AbortController | null = null

const INTERVALS: { value: ChartInterval; label: string; stepMs: number; defaultBars: number }[] = [
  { value: '5m', label: '5M', stepMs: 300_000, defaultBars: 156 },
  { value: '1h', label: '1H', stepMs: 3_600_000, defaultBars: 48 },
]

const STEP_BY_INTERVAL = Object.fromEntries(INTERVALS.map(i => [i.value, i.stepMs])) as Record<ChartInterval, number>
const DEFAULT_BARS = Object.fromEntries(INTERVALS.map(i => [i.value, i.defaultBars])) as Record<ChartInterval, number>

/** Equal-spaced x-axis — removes overnight/weekend visual gaps; real times live in tooltipDates. */
const SYNTHETIC_BASE_MS = Date.UTC(2020, 0, 1)

const barCount = computed(() => pricePoints.value.length)
const forecastBarCount = computed(() => Math.max(0, forecastPoints.value.length - 1))
const hasData = computed(() => pricePoints.value.length > 0)
const chartKey = computed(() => `${props.prediction.symbol}-${interval.value}-${props.reloadToken ?? 0}`)

function synthX(index: number, stepMs: number): number {
  return SYNTHETIC_BASE_MS + index * stepMs
}

function indexFromSynthX(x: number, stepMs: number): number {
  return Math.round((x - SYNTHETIC_BASE_MS) / stepMs)
}

function labelForSynthX(x: number): string {
  const idx = indexFromSynthX(x, axisStepMs.value)
  const date = tooltipDates.value[idx]
  if (!date) return ''
  const showDate = interval.value === '1h' || idx === 0 || idx === tooltipDates.value.length - 1
  return formatChartIntradayAxis(date, showDate)
}

function tooltipForSynthX(x: number): string {
  const idx = indexFromSynthX(x, axisStepMs.value)
  const date = tooltipDates.value[idx]
  return date ? formatChartIntradayDetail(date) : ''
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
  return computeBounds([...inRange(pricePoints.value), ...inRange(forecastPoints.value)])
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

function defaultXRange(stepMs: number, priceBars: number, forecastBars: number): { min: number; max: number } {
  const window = Math.min(DEFAULT_BARS[interval.value], priceBars)
  const startIdx = Math.max(0, priceBars - window)
  const endIdx = priceBars - 1 + forecastBars
  return {
    min: synthX(startIdx, stepMs) - stepMs * 0.5,
    max: synthX(endIdx, stepMs) + stepMs * 0.5,
  }
}

function fitDefaultView(chart: ChartRef) {
  const priceBars = pricePoints.value.length
  if (!priceBars) return
  const { min, max } = defaultXRange(axisStepMs.value, priceBars, forecastBarCount.value)
  chart.zoomX(min, max)
  fitYAxis(chart, min, max)
}

function buildChartData(
  history: ChartPoint[],
  forecast: ChartPoint[],
  iv: ChartInterval,
) {
  const stepMs = STEP_BY_INTERVAL[iv]
  axisStepMs.value = stepMs

  const dates: string[] = []
  const prices: LinePoint[] = []

  for (let i = 0; i < history.length; i++) {
    const close = roundPrice(history[i]!.close)
    if (close == null) continue
    dates[i] = history[i]!.date
    prices.push({ x: synthX(i, stepMs), y: close })
  }

  const forecastLine: LinePoint[] = []
  let markerX: number | null = null
  const lastPoint = prices.at(-1)

  if (lastPoint && forecast.length) {
    const bridgeIdx = prices.length - 1
    markerX = lastPoint.x
    forecastLine.push(lastPoint)

    for (let i = 0; i < forecast.length; i++) {
      const close = roundPrice(forecast[i]!.close)
      if (close == null) continue
      const idx = bridgeIdx + 1 + i
      dates[idx] = forecast[i]!.date
      forecastLine.push({ x: synthX(idx, stepMs), y: close })
    }
  }

  tooltipDates.value = dates
  pricePoints.value = prices
  forecastPoints.value = forecastLine
  forecastMarkerX.value = markerX
}

const chartSeries = computed<ApexOptions['series']>(() => {
  const series: ApexOptions['series'] = [
    { name: 'Price', type: 'line', data: pricePoints.value },
  ]
  if (forecastPoints.value.length) {
    series.push({
      name: `Forecast (${forecastEngine.value ?? 'baseline'})`,
      type: 'line',
      data: forecastPoints.value,
    })
  }
  return series
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
    width: [2, 2.5],
    curve: 'straight',
    dashArray: [0, 6],
  },
  colors: ['#2563eb', '#f97316'],
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
      formatter: (val: string) => labelForSynthX(Number(val)),
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
    show: forecastPoints.value.length > 0,
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
    x: { show: true, formatter: (val: number) => tooltipForSynthX(val) },
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

async function loadChart() {
  fetchAbort?.abort()
  fetchAbort = new AbortController()

  loading.value = true
  loadError.value = null

  const iv = interval.value
  const symbol = props.prediction.symbol

  try {
    const res = await api.fetchChartSeries(symbol, iv)
    if (fetchAbort.signal.aborted) return
    buildChartData(res.points, res.forecast?.points ?? [], iv)
    chartMeta.value = res.meta
    forecastEngine.value = res.forecast?.engine ?? res.meta.forecast_engine ?? null
  } catch (e: unknown) {
    if (fetchAbort.signal.aborted) return
    const raw = e instanceof Error ? e.message : 'Failed to load chart data'
    loadError.value = raw.toLowerCase().includes('timeout') || raw.toLowerCase().includes('aborted')
      ? 'Chart request timed out. Retry shortly or check /tmp/radar-api.log.'
      : raw
    pricePoints.value = []
    forecastPoints.value = []
    tooltipDates.value = []
    chartMeta.value = null
    forecastEngine.value = null
  } finally {
    if (!fetchAbort.signal.aborted) {
      loading.value = false
    }
  }
}

watch(interval, loadChart)
watch(() => props.prediction.symbol, loadChart, { immediate: true })
watch(() => props.reloadToken, () => {
  loadChart()
})

watch(
  () => [pricePoints.value.length, forecastPoints.value.length],
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
        <div v-if="chartMeta" class="text-muted-foreground">
          {{ chartMeta.rows }} bars · {{ chartMeta.forecast_bars ?? 0 }} forecast · {{ forecastEngine ?? chartMeta.forecast_engine }}
        </div>
      </div>

      <UiTabs v-model="interval" class="w-full sm:w-auto">
        <UiTabsList class="grid w-full grid-cols-2 sm:w-auto">
          <UiTabsTrigger v-for="item in INTERVALS" :key="item.value" :value="item.value">
            {{ item.label }}
          </UiTabsTrigger>
        </UiTabsList>
      </UiTabs>
    </div>

    <p class="text-xs text-muted-foreground">
      Times shown in Europe/Berlin · trading hours 10:00–02:00 only · bars evenly spaced (no overnight/weekend gaps)
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
        <span class="text-sm text-muted-foreground">Loading {{ interval.toUpperCase() }} data + baseline path…</span>
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
