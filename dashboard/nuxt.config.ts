import tailwindcss from "@tailwindcss/vite";

export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: true },

  runtimeConfig: {
    radarRoot: process.env.RADAR_ROOT || '',
    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE || '',
    },
  },

  nitro: {
    devProxy: {
      '/api/predictions': { target: 'http://127.0.0.1:8000/api/predictions', changeOrigin: true },
      '/api/refresh': { target: 'http://127.0.0.1:8000/api/refresh', changeOrigin: true },
      '/api/bootstrap': { target: 'http://127.0.0.1:8000/api/bootstrap', changeOrigin: true, timeout: 600_000 },
      '/api/chart': {
        target: 'http://127.0.0.1:8000/api/chart',
        changeOrigin: true,
        timeout: 600_000,
      },
      '/api/performance': { target: 'http://127.0.0.1:8000/api/performance', changeOrigin: true },
      '/api/news': { target: 'http://127.0.0.1:8000/api/news', changeOrigin: true },
      '/api/meta': { target: 'http://127.0.0.1:8000/api/meta', changeOrigin: true },
      '/health': { target: 'http://127.0.0.1:8000/health', changeOrigin: true },
    },
  },

  modules: [
    '@nuxtjs/color-mode',
    'motion-v/nuxt',
    '@vueuse/nuxt',
    '@nuxt/icon',
    '@nuxt/fonts'
  ],

  imports: {
    imports: [{
      from: 'tailwind-variants',
      name: 'tv'
    }, {
      from: 'tailwind-variants',
      name: 'VariantProps',
      type: true
    }]
  },

  colorMode: {
    storageKey: 'dashboard-color-mode',
    classSuffix: ''
  },

  icon: {
    clientBundle: {
      scan: true,
      sizeLimitKb: 0
    },

    mode: 'svg',
    class: 'shrink-0',
    fetchTimeout: 2000,
    serverBundle: 'local'
  },

  css: ['~/assets/css/tailwind.css'],

  vite: {
    plugins: [tailwindcss()]
  }
})
