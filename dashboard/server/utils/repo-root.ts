import { existsSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export function repoRoot(): string {
  const fromEnv = process.env.RADAR_ROOT
  if (fromEnv && existsSync(join(fromEnv, 'scripts/ensure-api.sh'))) {
    return resolve(fromEnv)
  }

  const here = dirname(fileURLToPath(import.meta.url))
  const candidates = [
    resolve(here, '../../..'),
    resolve(process.cwd(), '..'),
    resolve(process.cwd()),
  ]

  for (const candidate of candidates) {
    if (existsSync(join(candidate, 'scripts/ensure-api.sh'))) {
      return candidate
    }
  }

  throw createError({
    statusCode: 500,
    statusMessage: 'Could not locate TradingBot repo root (scripts/ensure-api.sh)',
  })
}
