<script setup lang="ts">
import type { Prediction } from '~/types/radar'
import { formatPrice, formatSentimentPercent, sentimentPercentClass } from '~/utils/format'
import {
  signalMeterLabel,
  signalMeterPercent,
  signalMeterValue,
  noTradeReasons,
  noTradeReasonsSummary,
  tradeActionHint,
  tradeActionLabel,
} from '~/utils/signalLabels'

const props = defineProps<{
  prediction: Prediction
  selected?: boolean
}>()

const emit = defineEmits<{
  select: [symbol: string]
}>()

const isBuy = computed(() => props.prediction.signal === 1)
const blockedReasons = computed(() => noTradeReasons(props.prediction))
const badgeVariant = computed(() => {
  if (props.prediction.confidence === 'high') return 'default'
  if (props.prediction.confidence === 'medium') return 'secondary'
  return 'outline'
})
</script>

<template>
  <UiCard
    class="cursor-pointer transition-all hover:shadow-md"
    :class="selected ? 'ring-2 ring-blue-500' : ''"
    @click="emit('select', prediction.symbol)"
  >
    <UiCardHeader>
      <div class="flex items-center justify-between gap-2">
        <UiCardTitle>{{ prediction.symbol }}</UiCardTitle>
        <UiBadge :variant="isBuy ? 'default' : 'outline'">
          {{ tradeActionLabel(prediction) }}
        </UiBadge>
      </div>
      <UiCardDescription :title="isBuy ? tradeActionHint(prediction) : noTradeReasonsSummary(prediction)">
        <template v-if="isBuy">
          Actionable buy — passes all gates
        </template>
        <template v-else>
          <span class="block">AI return ≠ buy signal (gates use P(up) + filters).</span>
          <span v-if="blockedReasons.length" class="mt-1 block text-xs">
            {{ blockedReasons.join(' · ') }}
          </span>
          <span v-else class="mt-1 block text-xs text-muted-foreground">
            Refresh predictions to see gate breakdown (API update).
          </span>
        </template>
      </UiCardDescription>
    </UiCardHeader>
    <UiCardContent class="space-y-3">
      <div class="grid grid-cols-2 gap-3 text-sm">
        <div>
          <p class="text-muted-foreground">Last close</p>
          <p class="font-semibold tabular-nums">{{ formatPrice(prediction.last_close) }}</p>
        </div>
        <div>
          <p class="text-muted-foreground">P(up)</p>
          <p class="font-semibold">{{ ((prediction.p_up ?? 0) * 100).toFixed(1) }}%</p>
        </div>
        <div>
          <p class="text-muted-foreground">Confluence</p>
          <p class="font-semibold">{{ ((prediction.confluence_score ?? 0) * 100).toFixed(0) }}%</p>
        </div>
        <div title="Ensemble return model — predicted 1d return">
          <p class="text-muted-foreground">AI return (1d)</p>
          <p class="font-semibold" :class="(prediction.forecast_return_1d ?? 0) >= 0 ? 'text-emerald-600' : 'text-red-500'">
            {{ ((prediction.forecast_return_1d ?? 0) * 100).toFixed(2) }}%
          </p>
        </div>
        <div v-if="prediction.entry_quality != null">
          <p class="text-muted-foreground">Entry timing</p>
          <p class="font-semibold">{{ ((prediction.entry_quality ?? 0) * 100).toFixed(0) }}%</p>
        </div>
        <div v-if="prediction.position_size != null && prediction.signal === 1">
          <p class="text-muted-foreground">Position size</p>
          <p class="font-semibold">{{ ((prediction.position_size ?? 0) * 100).toFixed(1) }}%</p>
        </div>
        <div v-if="prediction.sentiment_mean != null">
          <p class="text-muted-foreground">News sentiment</p>
          <p class="font-semibold" :class="sentimentPercentClass(prediction.sentiment_mean)">
            {{ formatSentimentPercent(prediction.sentiment_mean) }}
          </p>
        </div>
      </div>
      <div class="space-y-1">
        <div class="flex justify-between text-xs text-muted-foreground">
          <span>{{ signalMeterLabel(prediction) }}</span>
          <span class="font-medium text-foreground">{{ signalMeterValue(prediction) }}</span>
        </div>
        <UiProgress :model-value="signalMeterPercent(prediction)" />
      </div>
    </UiCardContent>
  </UiCard>
</template>
