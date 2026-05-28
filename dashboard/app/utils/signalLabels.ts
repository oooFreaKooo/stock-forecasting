import type { Prediction } from '~/types/radar'

/** User-facing trade action (not a loading state). */
export function tradeActionLabel(prediction: Prediction): string {
  if (prediction.action === 'BUY' || prediction.signal === 1) return 'BUY'
  if (prediction.action === 'NO TRADE' || prediction.action === 'WAIT') return 'No trade'
  return prediction.action ?? 'No trade'
}

export function tradeActionHint(prediction: Prediction): string {
  if (prediction.signal === 1) {
    return 'Passes ensemble gates — actionable buy signal'
  }
  return 'Model output only — does not pass buy filters (probability, confluence, momentum, etc.)'
}

/** Bar label: confidence applies only when there is a BUY signal. */
export function signalMeterLabel(prediction: Prediction): string {
  return prediction.signal === 1 ? 'Confidence' : 'Confluence'
}

export function signalMeterValue(prediction: Prediction): string {
  if (prediction.signal === 1) {
    const level = prediction.confidence
    if (level === 'high') return 'High'
    if (level === 'medium') return 'Medium'
    if (level === 'low') return 'Low'
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
