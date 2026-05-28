<script setup lang="ts">
import type { ApiStatus, NewsSnapshot, PerformanceMetrics, Prediction } from '~/types/radar'

const api = useRadarApi()

const refreshing = ref(false)
const error = ref<string | null>(null)
const predictions = ref<Prediction[]>([])
const metrics = ref<PerformanceMetrics | null>(null)
const news = ref<NewsSnapshot | null>(null)
const apiStatus = ref<ApiStatus | null>(null)
const apiStarting = ref(false)
const selectedSymbol = ref('AAPL')
const lastFetched = ref<string | null>(null)
const chartReloadToken = ref(0)

let autoStartTimer: ReturnType<typeof setInterval> | null = null

const apiOnline = computed(() => apiStatus.value?.online === true)

const apiNeedsReload = computed(() =>
  apiOnline.value && (apiStatus.value?.news_enabled === false || apiStatus.value?.stale === true),
)

const apiStatusLabel = computed(() => {
  if (apiStarting.value) return 'API starting…'
  if (!apiStatus.value?.online) return 'API offline'
  if (apiNeedsReload.value) return 'API reloading…'
  return `API v${apiStatus.value.version ?? '?'}`
})

const selectedPrediction = computed(() =>
  predictions.value.find(p => p.symbol === selectedSymbol.value) ?? predictions.value[0] ?? null,
)

async function waitForApiOnline(maxAttempts = 80) {
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    await loadApiStatus()
    if (apiOnline.value && !apiNeedsReload.value) return true
    await new Promise(resolve => setTimeout(resolve, 250))
  }
  return false
}

async function loadCachedState() {
  if (!apiOnline.value) return
  try {
    const [predRes, newsRes] = await Promise.all([
      api.fetchPredictions(),
      api.fetchNews(),
    ])
    predictions.value = predRes.predictions.filter(p => !p.error)
    news.value = newsRes
    try {
      metrics.value = await api.fetchPerformance()
    } catch {
      // backtest metrics optional until ensemble is trained
    }
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : 'Failed to load dashboard'
  }
}

async function refreshAll() {
  if (!apiOnline.value) return
  refreshing.value = true
  error.value = null
  try {
    const res = await api.refreshDashboard()
    predictions.value = res.predictions.filter(p => !p.error)
    news.value = res.news
    metrics.value = res.metrics
    lastFetched.value = new Date().toISOString()
    chartReloadToken.value += 1
    if (!predictions.value.find(p => p.symbol === selectedSymbol.value)) {
      selectedSymbol.value = predictions.value[0]?.symbol ?? 'AAPL'
    }
  } catch (e: unknown) {
    const raw = e instanceof Error ? e.message : 'Refresh failed'
    error.value = raw.toLowerCase().includes('timeout')
      ? 'Refresh timed out. The first run after a long gap can take a few minutes.'
      : raw
  } finally {
    refreshing.value = false
  }
}

async function loadApiStatus() {
  try {
    apiStatus.value = await api.fetchApiStatus()
  } catch {
    apiStatus.value = { online: false, version: null, started_at: null, news_enabled: false, features: [], routes: [] }
  }
}

async function ensureApiRunning() {
  if (apiStarting.value) return apiOnline.value

  await loadApiStatus()
  if (apiOnline.value && !apiNeedsReload.value) {
    return true
  }

  apiStarting.value = true
  error.value = null
  try {
    await api.ensureApi(apiNeedsReload.value)
    const ready = await waitForApiOnline()
    if (!ready) {
      error.value = 'API did not start. Check /tmp/radar-api.log'
      return false
    }
    await loadCachedState()
    return true
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : 'Failed to start API'
    return false
  } finally {
    apiStarting.value = false
  }
}

onMounted(async () => {
  await ensureApiRunning()
  autoStartTimer = setInterval(async () => {
    if (!apiOnline.value || apiNeedsReload.value) {
      await ensureApiRunning()
    }
  }, 8_000)
})

onUnmounted(() => {
  if (autoStartTimer) clearInterval(autoStartTimer)
})
</script>

<template>
  <UiSidebarProvider :default-open="true" class="min-h-svh w-full overflow-x-hidden [--sidebar-width:22rem]">
    <UiSidebarInset class="min-w-0">
      <header class="sticky top-0 z-20 border-b bg-card/80 backdrop-blur">
        <div class="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4 sm:flex-row sm:items-center sm:justify-between">
          <div class="flex items-start gap-3">
            <UiSidebarTrigger icon="lucide:newspaper" label="Toggle news feed" class="mt-1 shrink-0" />
            <div>
              <h1 class="text-2xl font-bold tracking-tight">Hybrid AI Investment Radar</h1>
              <p class="text-sm text-muted-foreground">
                Ensemble direction model · daily signals · live intraday charts
              </p>
              <div class="mt-2 flex flex-wrap items-center gap-2">
                <UiBadge
                  :variant="apiOnline && !apiNeedsReload ? 'outline' : 'destructive'"
                >
                  {{ apiStatusLabel }}
                </UiBadge>
                <span v-if="lastFetched" class="text-xs text-muted-foreground">
                  data refreshed {{ new Date(lastFetched).toLocaleString() }}
                </span>
                <span v-else-if="apiStatus?.started_at" class="text-xs text-muted-foreground">
                  API started {{ new Date(apiStatus.started_at).toLocaleString() }}
                </span>
              </div>
            </div>
          </div>
          <div class="flex flex-wrap gap-2">
            <UiButton
              variant="default"
              :loading="refreshing || apiStarting"
              :disabled="!apiOnline && !apiStarting"
              icon="lucide:refresh-cw"
              text="Refresh Data & Predictions"
              @click="refreshAll"
            />
          </div>
        </div>
      </header>

      <main class="mx-auto max-w-7xl space-y-6 px-4 py-6">
        <UiCard v-if="error" class="border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-950/30">
          <UiCardContent class="pt-6 text-sm text-red-700 dark:text-red-300">
            {{ error }}
            <p class="mt-2 text-xs text-muted-foreground">
              Refresh fetches latest prices, rebuilds features, and rescored signals. Logs:
              <code class="rounded bg-black/10 px-1">/tmp/radar-api.log</code>
            </p>
          </UiCardContent>
        </UiCard>

        <RadarStatsBar :metrics="metrics" :loading="refreshing || apiStarting" />

        <UiCard>
          <UiCardHeader class="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <UiCardTitle class="text-xl">
                {{ selectedPrediction?.symbol ?? '—' }} — Prediction Chart
              </UiCardTitle>
              <UiCardDescription>
                Live 5M/1H prices with short-term forecast overlay
              </UiCardDescription>
            </div>
            <div class="flex flex-wrap gap-2 text-sm">
              <UiBadge variant="outline">5M / 1H</UiBadge>
              <UiBadge variant="outline">Scroll zoom</UiBadge>
              <UiBadge variant="outline">Drag pan</UiBadge>
            </div>
          </UiCardHeader>
          <UiCardContent class="pb-2">
            <RadarPredictionChart
              v-if="selectedPrediction"
              :prediction="selectedPrediction"
              :reload-token="chartReloadToken"
            />
            <p v-else-if="apiStarting" class="text-sm text-muted-foreground">Waiting for API…</p>
            <p v-else class="text-sm text-muted-foreground">Select a symbol or refresh to load predictions.</p>
          </UiCardContent>
        </UiCard>

        <section>
          <h2 class="mb-4 text-lg font-semibold">Symbol Signals</h2>
          <div class="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            <RadarSymbolCard
              v-for="pred in predictions"
              :key="pred.symbol"
              :prediction="pred"
              :selected="pred.symbol === selectedSymbol"
              @select="selectedSymbol = $event"
            />
          </div>
        </section>
      </main>
    </UiSidebarInset>

    <RadarNewsSidebar
      :news="news"
      :selected-symbol="selectedSymbol"
      :loading="refreshing"
      @refresh="refreshAll"
    />
  </UiSidebarProvider>
</template>
