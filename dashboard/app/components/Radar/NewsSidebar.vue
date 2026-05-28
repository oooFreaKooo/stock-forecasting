<script setup lang="ts">
import type { NewsHeadline, NewsSnapshot } from '~/types/radar'

const props = defineProps<{
  news: NewsSnapshot | null
  selectedSymbol?: string
  loading?: boolean
}>()

const emit = defineEmits<{
  refresh: []
}>()

const filteredHeadlines = computed(() => {
  const headlines = props.news?.headlines ?? []
  if (!props.selectedSymbol) return headlines
  return headlines.filter(h => h.symbol === props.selectedSymbol || h.symbol === 'MARKET')
})

const selectedSymbolNews = computed(() => {
  if (!props.selectedSymbol || !props.news?.symbols) return null
  return props.news.symbols[props.selectedSymbol] ?? null
})

function sentimentLabel(value: number) {
  if (value >= 0.25) return 'Bullish'
  if (value <= -0.25) return 'Bearish'
  return 'Neutral'
}

function sentimentClass(value: number) {
  if (value >= 0.25) return 'text-emerald-600 dark:text-emerald-400'
  if (value <= -0.25) return 'text-red-500 dark:text-red-400'
  return 'text-muted-foreground'
}

function formatHeadlineTime(headline: NewsHeadline) {
  const raw = headline.published ?? headline.date
  const date = new Date(raw)
  if (Number.isNaN(date.getTime())) return raw
  return date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}
</script>

<template>
  <UiSidebar side="right" collapsible="offcanvas" class="border-l">
    <UiSidebarHeader class="border-b border-sidebar-border">
      <div class="flex items-start justify-between gap-2 px-1">
        <div class="min-w-0">
          <p class="text-sm font-semibold">News Feed</p>
          <p class="text-xs text-muted-foreground">
            Yahoo RSS · VADER sentiment
          </p>
        </div>
        <UiButton
          variant="ghost"
          size="icon-sm"
          icon="lucide:refresh-cw"
          :loading="loading"
          text="Refresh news"
          @click="emit('refresh')"
        />
      </div>

      <div v-if="news?.enabled" class="grid grid-cols-2 gap-2 px-1">
        <div class="rounded-md border border-sidebar-border bg-sidebar-accent/40 p-2">
          <p class="text-[10px] uppercase tracking-wide text-muted-foreground">Market</p>
          <p class="text-sm font-semibold" :class="sentimentClass(news.market_sentiment ?? 0)">
            {{ sentimentLabel(news.market_sentiment ?? 0) }}
          </p>
        </div>
        <div class="rounded-md border border-sidebar-border bg-sidebar-accent/40 p-2">
          <p class="text-[10px] uppercase tracking-wide text-muted-foreground">Headlines</p>
          <p class="text-sm font-semibold tabular-nums">{{ news.headline_count ?? 0 }}</p>
        </div>
      </div>
    </UiSidebarHeader>

    <UiSidebarContent>
      <UiScrollArea class="h-full px-2">
        <div class="space-y-3 py-2">
          <div v-if="loading" class="px-2 text-sm text-muted-foreground">
            Fetching headlines and recalculating forecasts...
          </div>

          <div v-else-if="!news?.enabled" class="px-2 text-sm text-muted-foreground">
            News integration is disabled in config.
          </div>

          <template v-else>
            <div
              v-if="selectedSymbolNews"
              class="rounded-md border border-sidebar-border bg-sidebar-accent/30 p-2"
            >
              <p class="text-[10px] uppercase tracking-wide text-muted-foreground">
                {{ selectedSymbol }} sentiment
              </p>
              <p class="text-sm font-semibold" :class="sentimentClass(selectedSymbolNews.sentiment_mean)">
                {{ sentimentLabel(selectedSymbolNews.sentiment_mean) }}
                <span class="text-xs font-normal text-muted-foreground">
                  ({{ (selectedSymbolNews.sentiment_mean * 100).toFixed(0) }})
                </span>
              </p>
            </div>

            <UiSeparator />

            <div v-if="filteredHeadlines.length === 0" class="px-2 text-sm text-muted-foreground">
              No headlines for {{ selectedSymbol ?? 'the current universe' }}.
            </div>

            <ul v-else class="space-y-2">
              <li
                v-for="(headline, index) in filteredHeadlines"
                :key="`${headline.symbol}-${index}-${headline.title}`"
                class="rounded-md border border-sidebar-border bg-background/60 p-3"
              >
                <div class="mb-1 flex items-center justify-between gap-2">
                  <UiBadge variant="outline" class="text-[10px]">{{ headline.symbol }}</UiBadge>
                  <span class="text-[10px] font-medium tabular-nums" :class="sentimentClass(headline.sentiment)">
                    {{ headline.sentiment >= 0 ? '+' : '' }}{{ (headline.sentiment * 100).toFixed(0) }}
                  </span>
                </div>
                <p class="text-sm leading-snug">{{ headline.title }}</p>
                <p class="mt-1 text-[10px] text-muted-foreground">{{ formatHeadlineTime(headline) }}</p>
              </li>
            </ul>
          </template>
        </div>
      </UiScrollArea>
    </UiSidebarContent>

    <UiSidebarFooter class="border-t border-sidebar-border">
      <p v-if="news?.fetched_at" class="px-2 text-[10px] text-muted-foreground">
        Updated {{ new Date(news.fetched_at).toLocaleString() }}
      </p>
      <p v-else class="px-2 text-[10px] text-muted-foreground">
        Click refresh to load headlines
      </p>
    </UiSidebarFooter>
  </UiSidebar>
</template>
