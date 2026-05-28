<script setup lang="ts">
import type { PerformanceMetrics } from '~/types/radar'

defineProps<{
  metrics: PerformanceMetrics | null
  loading?: boolean
}>()
</script>

<template>
  <div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-6">
    <UiCard>
      <UiCardHeader>
        <UiCardDescription>OOS hit rate</UiCardDescription>
        <UiCardTitle class="text-2xl">
          {{ metrics ? `${(metrics.gated_hit_rate * 100).toFixed(1)}%` : '—' }}
        </UiCardTitle>
      </UiCardHeader>
      <UiCardContent>
        <p class="text-sm text-muted-foreground">{{ metrics?.gated_trades ?? 0 }} backtest trades</p>
      </UiCardContent>
    </UiCard>

    <UiCard>
      <UiCardHeader>
        <UiCardDescription>Expectancy E[R]</UiCardDescription>
        <UiCardTitle class="text-2xl">
          {{ metrics?.expectancy != null ? `${(metrics.expectancy * 100).toFixed(2)}%` : '—' }}
        </UiCardTitle>
      </UiCardHeader>
      <UiCardContent>
        <p class="text-sm text-muted-foreground">Per trade after costs</p>
      </UiCardContent>
    </UiCard>

    <UiCard>
      <UiCardHeader>
        <UiCardDescription>Profit factor</UiCardDescription>
        <UiCardTitle class="text-2xl">
          {{ metrics?.profit_factor != null ? metrics.profit_factor.toFixed(2) : '—' }}
        </UiCardTitle>
      </UiCardHeader>
      <UiCardContent>
        <p class="text-sm text-muted-foreground">Gross wins / gross losses</p>
      </UiCardContent>
    </UiCard>

    <UiCard>
      <UiCardHeader>
        <UiCardDescription>Max drawdown</UiCardDescription>
        <UiCardTitle class="text-2xl">
          {{ metrics?.max_drawdown != null ? `${(metrics.max_drawdown * 100).toFixed(1)}%` : '—' }}
        </UiCardTitle>
      </UiCardHeader>
      <UiCardContent>
        <p class="text-sm text-muted-foreground">OOS gated equity curve</p>
      </UiCardContent>
    </UiCard>

    <UiCard>
      <UiCardHeader>
        <UiCardDescription>Paper trading</UiCardDescription>
        <UiCardTitle class="text-2xl">
          {{ metrics?.paper_trading?.n_resolved ? `${(metrics.paper_trading.hit_rate * 100).toFixed(1)}%` : '—' }}
        </UiCardTitle>
      </UiCardHeader>
      <UiCardContent>
        <p class="text-sm text-muted-foreground">{{ metrics?.paper_trading?.n_resolved ?? 0 }} resolved signals</p>
      </UiCardContent>
    </UiCard>

    <UiCard>
      <UiCardHeader>
        <UiCardDescription>Signal threshold</UiCardDescription>
        <UiCardTitle class="text-2xl">
          {{ metrics ? metrics.threshold_used.toFixed(2) : '—' }}
        </UiCardTitle>
      </UiCardHeader>
      <UiCardContent>
        <p class="text-sm text-muted-foreground">Minimum P(up) for BUY</p>
      </UiCardContent>
    </UiCard>
  </div>
</template>
