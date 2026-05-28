export interface ChartPoint {
  date: string
  close: number
}

export type ChartInterval = '5m' | '1h' | '1d'

export interface ChartForecast {
  engine: string
  horizon_bars: number
  points: ChartPoint[]
}

export interface ChartValidationMetrics {
  mae: number | null
  mape: number | null
  rmse: number | null
  direction_accuracy: number | null
  n_points: number
  validation_days?: number
  validation_bars?: number
  anchor_date?: string
}

export interface ChartValidation {
  engine: string
  points: ChartPoint[]
  metrics: ChartValidationMetrics
}

export interface ChartModelPath {
  engine: string
  points: ChartPoint[]
  backtest_bars?: number
  forward_bars?: number
}

export interface ChartSeriesResponse {
  symbol: string
  interval: ChartInterval
  points: ChartPoint[]
  model?: ChartModelPath
  forecast: ChartForecast
  validation?: ChartValidation
  meta: {
    source: string
    rows: number
    note: string
    period?: string
    limit?: number
    forecast_engine?: string
    forecast_bars?: number
    validation_bars?: number
    display_timezone?: string
    ai_p_up?: number
    ai_return_1d?: number | null
    ai_target_price_1d?: number | null
  }
}

export interface ChartBundleResponse {
  symbol: string
  intraday: ChartSeriesResponse
  /** ~30d hourly actuals; AI overlay resampled from intraday (5m). */
  intraday_1h: ChartSeriesResponse
  daily: ChartSeriesResponse
}

export interface NewsHeadline {
  symbol: string
  title: string
  sentiment: number
  published?: string | null
  date: string
}

export interface SymbolNewsStats {
  sentiment_mean: number
  sentiment_ma: number
  headline_count: number
  as_of_date: string
}

export interface NewsSnapshot {
  enabled: boolean
  fetched_at?: string
  headline_count?: number
  market_sentiment?: number
  market_sentiment_dispersion?: number
  symbols?: Record<string, SymbolNewsStats>
  headlines?: NewsHeadline[]
}

export interface PredictionGates {
  probability?: boolean
  forecast?: boolean
  memory?: boolean
  event?: boolean
  agreement?: boolean
  horizon?: boolean
  momentum?: boolean
  vol?: boolean
  confluence?: boolean
}

export interface Prediction {
  symbol: string
  date?: string
  last_close?: number
  p_up?: number
  forecast_return_1d?: number
  signal?: number
  confidence?: string
  confluence_score?: number
  gates?: PredictionGates
  probability_threshold?: number
  action?: string
  sentiment_mean?: number
  headline_count?: number
  market_sentiment?: number
  news_fetched_at?: string
  probability_source?: string
  entry_quality?: number
  position_size?: number
  predicted_return_1d?: number
  portfolio_blocked?: boolean
  error?: string
}

export interface PerformanceMetrics {
  threshold_used: number
  optimized_params?: Record<string, unknown>
  simple_hit_rate: number
  simple_trades: number
  gated_v1_hit_rate: number
  gated_v1_trades: number
  gated_hit_rate: number
  gated_trades: number
  coverage_pct: number
  expectancy?: number
  profit_factor?: number
  max_drawdown?: number
  paper_trading?: {
    n_logged: number
    n_resolved: number
    hit_rate: number
    avg_return: number
  }
}

export interface PredictionsResponse {
  predictions: Prediction[]
  strategy: string
}

export interface DashboardRefreshResponse extends PredictionsResponse {
  fetched: Record<string, number>
  news: NewsSnapshot
  metrics: PerformanceMetrics | null
}

export interface BootstrapResponse extends PredictionsResponse {
  status: 'cached' | 'computed'
  cached_at?: string
  news: NewsSnapshot
  metrics?: PerformanceMetrics | null
}

export interface ApiStatus {
  online: boolean
  version: string | null
  started_at: string | null
  news_enabled: boolean
  features: string[]
  routes: string[]
  stale?: boolean
  predictions_cached?: boolean
  predictions_cached_at?: string | null
}

export interface ApiEnsureResponse {
  ok: boolean
  status: string
  force?: boolean
  message: string
}
