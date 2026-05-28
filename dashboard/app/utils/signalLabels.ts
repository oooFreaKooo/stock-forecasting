import type { Prediction } from '~/types/radar'

function normAction(action: string | undefined): string {
  return (action ?? '').trim().toUpperCase()
}

/** User-facing trade action (not a loading state). */
export function tradeActionLabel(prediction: Prediction): string {
  if (prediction.signal === 1) return 'BUY'
  const action = normAction(prediction.action)
  if (action === 'BUY') return 'BUY'
  if (action === 'NO TRADE' || action === 'WAIT' || action === '') return 'No trade'
  return prediction.action ?? 'No trade'
}

export function tradeActionHint(prediction: Prediction): string {
  if (prediction.signal === 1) {
    return 'Passes ensemble gates — actionable buy signal'
  }
  if (prediction.portfolio_blocked) {
    return 'Passed buy gates but ranked below portfolio top-N — not traded today'
  }
  const action = normAction(prediction.action)
  if (action === 'WAIT') {
    return 'Internal WAIT: demoted by portfolio limits or no buy signal — not “loading”'
  }
  return 'Model output only — does not pass buy filters (probability, confluence, momentum, etc.)'
}

/** Bar label: confidence applies only when there is a BUY signal. */
export function signalMeterLabel(prediction: Prediction): string {
  return prediction.signal === 1 ? 'Confidence' : 'Confluence'
}

export function signalMeterValue(prediction: Prediction): string {
  if (prediction.signal === 1) {
    const level = (prediction.confidence ?? '').toLowerCase()
    if (level === 'high') return 'High'
    if (level === 'medium') return 'Medium'
    if (level === 'low') return 'Low'
    if (level === 'none' || level === '') return '—'
    return '—'
  }
  const score = prediction.confluence_score ?? prediction.p_up ?? 0
  return `${Math.round(score * 100)}%`
}

export function signalMeterPercent(prediction: Prediction): number {
  if (prediction.signal === 1) {
    const level = prediction.confidence
    if (level === 'high') return 85
    if (level === 'medium') return 70
    if (level === 'low') return 55
    return (prediction.confluence_score ?? prediction.p_up ?? 0) * 100
  }
  return (prediction.confluence_score ?? prediction.p_up ?? 0) * 100
}

const GATE_LABELS: Record<keyof NonNullable<Prediction['gates']>, string> = {
  probability: 'P(up) below threshold',
  forecast: 'AI 1d return not positive',
  memory: 'Memory neighbors weak',
  event: 'Macro event day',
  agreement: 'Models disagree',
  horizon: 'Multi-horizon filter',
  momentum: 'Momentum rank too low',
  vol: 'Volatility regime too high',
  confluence: 'Confluence too low',
}

/** Why “No trade” despite a positive daily target — empty when BUY or gates unknown. */
export function noTradeReasons(prediction: Prediction): string[] {
  if (prediction.signal === 1) return []
  if (prediction.portfolio_blocked) {
    return ['Ranked below portfolio top-N (another symbol took the slot)']
  }
  const gates = prediction.gates
  if (!gates) return []

  const reasons: string[] = []
  const thresh = prediction.probability_threshold ?? 0.56
  const pUp = prediction.p_up ?? 0

  if (gates.probability === false) {
    reasons.push(`P(up) ${(pUp * 100).toFixed(1)}% < ${(thresh * 100).toFixed(0)}% required`)
  }
  for (const [key, label] of Object.entries(GATE_LABELS)) {
    if (key === 'probability') continue
    if (gates[key as keyof typeof gates] === false) {
      reasons.push(label)
    }
  }
  if (!reasons.length && (prediction.forecast_return_1d ?? 0) > 0) {
    reasons.push('Not in cross-sectional top-N for today (probability ranking)')
  }
  return reasons
}

export function noTradeReasonsSummary(prediction: Prediction): string {
  const reasons = noTradeReasons(prediction)
  if (!reasons.length) {
    return 'No buy today — AI return is informational; trade needs P(up) and all gates'
  }
  return reasons.join(' · ')
}
