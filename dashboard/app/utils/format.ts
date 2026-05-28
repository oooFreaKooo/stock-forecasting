/** Parse API/chart values that may arrive as strings with excess precision. */
export function parsePrice(value: number | string | null | undefined): number | null {
  if (value == null || value === '') return null
  const n = typeof value === 'number' ? value : Number.parseFloat(String(value))
  return Number.isFinite(n) ? n : null
}

/** Round for chart series — keeps enough precision without float noise. */
export function roundPrice(value: number | string | null | undefined, decimals = 4): number | null {
  const n = parsePrice(value)
  if (n == null) return null
  const factor = 10 ** decimals
  return Math.round(n * factor) / factor
}

/** Human-readable USD price (no trailing zero spam). */
export function formatPrice(value: number | string | null | undefined): string {
  const n = parsePrice(value)
  if (n == null) return '—'

  const abs = Math.abs(n)
  const fractionDigits = abs >= 1000 ? 0 : abs >= 100 ? 2 : abs >= 1 ? 2 : 4

  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  }).format(n)
}

/** Compact axis labels for dense y-axis ticks. */
export function formatPriceAxis(value: number | string | null | undefined): string {
  const n = parsePrice(value)
  if (n == null) return ''

  const abs = Math.abs(n)
  if (abs >= 1000) {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      notation: 'compact',
      maximumFractionDigits: 1,
    }).format(n)
  }

  return formatPrice(n)
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

/** Chart times always shown in Europe/Berlin to match TradingView DE setup. */
export const CHART_TIMEZONE = 'Europe/Berlin'

const BERLIN_TIME: Intl.DateTimeFormatOptions = {
  timeZone: CHART_TIMEZONE,
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
}

/** API timestamps are UTC; naive ISO strings are treated as UTC. */
export function parseUtcIso(iso: string): Date {
  const trimmed = iso.trim()
  if (trimmed.endsWith('Z') || /[+-]\d{2}:\d{2}$/.test(trimmed)) {
    return new Date(trimmed)
  }
  return new Date(`${trimmed}Z`)
}

export function formatDateShort(iso: string): string {
  const d = parseUtcIso(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleDateString('de-DE', {
    timeZone: CHART_TIMEZONE,
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

/** Detailed date label for chart tooltips. */
export function formatChartDateDetail(ts: number): string {
  const d = new Date(ts)
  if (Number.isNaN(d.getTime())) return '—'
  return d.toLocaleDateString('de-DE', {
    timeZone: CHART_TIMEZONE,
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

/** Intraday tooltip — Berlin date + 24h time. */
export function formatChartIntradayDetail(iso: string): string {
  const d = parseUtcIso(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString('de-DE', {
    timeZone: CHART_TIMEZONE,
    weekday: 'short',
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    ...BERLIN_TIME,
  })
}

/** Intraday axis label — Berlin 24h time. */
export function formatChartIntradayAxis(iso: string, showDate = false): string {
  const d = parseUtcIso(iso)
  if (Number.isNaN(d.getTime())) return iso
  if (showDate) {
    return d.toLocaleString('de-DE', {
      timeZone: CHART_TIMEZONE,
      day: '2-digit',
      month: 'short',
      ...BERLIN_TIME,
    })
  }
  return d.toLocaleTimeString('de-DE', BERLIN_TIME)
}

/** Compact axis label when showing a short date window. */
export function formatChartAxisDate(ts: number): string {
  const d = new Date(ts)
  if (Number.isNaN(d.getTime())) return '—'
  return d.toLocaleDateString('de-DE', {
    timeZone: CHART_TIMEZONE,
    month: 'short',
    day: 'numeric',
  })
}
