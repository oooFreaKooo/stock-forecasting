import type {
  ApiEnsureResponse,
  ApiStatus,
  ChartBundleResponse,
  BootstrapResponse,
  DashboardRefreshResponse,
  NewsSnapshot,
  PerformanceMetrics,
  Prediction,
  PredictionsResponse,
} from '~/types/radar'

export function useRadarApi() {
  const config = useRuntimeConfig()
  const base = config.public.apiBase as string
  const requestTimeoutMs = 8_000
  const refreshTimeoutMs = 180_000
  const bootstrapTimeoutMs = 120_000
  const chartTimeoutMs = 240_000

  function apiUrl(path: string) {
    return `${base}${path}`.replace('//', '/')
  }

  async function apiFetch<T>(path: string, options?: Parameters<typeof $fetch>[1]) {
    return await $fetch<T>(apiUrl(path), {
      timeout: requestTimeoutMs,
      ...options,
    })
  }

  async function fetchPredictions(): Promise<PredictionsResponse> {
    return await apiFetch('/api/predictions')
  }

  async function refreshDashboard(): Promise<DashboardRefreshResponse> {
    return await apiFetch('/api/refresh', {
      method: 'POST',
      timeout: refreshTimeoutMs,
    })
  }

  /** Cached predictions or score from disk artifacts (no full market refresh). */
  async function bootstrapDashboard(): Promise<BootstrapResponse> {
    return await apiFetch('/api/bootstrap', {
      method: 'POST',
      timeout: bootstrapTimeoutMs,
    })
  }

  /** One 5M AI run + daily series; frontend resamples to 1H without refetch. */
  async function fetchChartBundle(symbol: string): Promise<ChartBundleResponse> {
    return await apiFetch(`/api/chart/${symbol}/bundle`, {
      timeout: chartTimeoutMs,
    })
  }

  async function fetchPerformance(): Promise<PerformanceMetrics> {
    return await apiFetch('/api/performance')
  }

  async function fetchNews(refresh = false): Promise<NewsSnapshot> {
    return await apiFetch('/api/news', {
      query: refresh ? { refresh: true } : undefined,
    })
  }

  async function checkHealth(): Promise<{ status: string }> {
    return await apiFetch('/health')
  }

  async function fetchApiStatus(): Promise<ApiStatus> {
    return await $fetch('/dev/api-status', { timeout: requestTimeoutMs })
  }

  async function ensureApi(force = false): Promise<ApiEnsureResponse> {
    return await $fetch('/dev/ensure-api', {
      method: 'POST',
      body: { force },
      timeout: 5_000,
    })
  }

  return {
    fetchPredictions,
    refreshDashboard,
    bootstrapDashboard,
    fetchChartBundle,
    fetchPerformance,
    fetchNews,
    checkHealth,
    fetchApiStatus,
    ensureApi,
  }
}
