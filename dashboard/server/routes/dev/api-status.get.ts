export default defineEventHandler(async () => {
  const base = process.env.NUXT_PUBLIC_API_BASE || 'http://127.0.0.1:8000'

  try {
    const meta = await $fetch<{
      version?: string
      started_at?: string
      features?: string[]
      news_enabled?: boolean
      routes?: string[]
      predictions_cached?: boolean
      predictions_cached_at?: string | null
    }>(`${base}/api/meta`, { timeout: 4_000 })

    return {
      online: true,
      version: meta.version ?? 'unknown',
      started_at: meta.started_at ?? null,
      news_enabled: meta.news_enabled ?? meta.routes?.includes('/api/news') ?? false,
      features: meta.features ?? [],
      routes: meta.routes ?? [],
      predictions_cached: meta.predictions_cached ?? false,
      predictions_cached_at: meta.predictions_cached_at ?? null,
    }
  } catch {
    try {
      await $fetch(`${base}/health`, { timeout: 2_000 })
      return {
        online: true,
        version: 'legacy',
        started_at: null,
        news_enabled: false,
        features: [],
        routes: [],
        stale: true,
      }
    } catch {
      return {
        online: false,
        version: null,
        started_at: null,
        news_enabled: false,
        features: [],
        routes: [],
      }
    }
  }
})
