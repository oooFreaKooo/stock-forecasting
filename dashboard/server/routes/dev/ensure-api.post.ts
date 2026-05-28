import { spawn } from 'node:child_process'
import { join } from 'node:path'
import { repoRoot } from '../../utils/repo-root'

export default defineEventHandler(async (event) => {
  if (!import.meta.dev) {
    throw createError({ statusCode: 403, statusMessage: 'API auto-start is dev-only' })
  }

  const body = await readBody<{ force?: boolean }>(event).catch(() => ({}))
  const force = body?.force === true ? '1' : '0'
  const root = repoRoot()
  const script = join(root, 'scripts/ensure-api.sh')

  const child = spawn('bash', [script, force], {
    cwd: root,
    detached: true,
    stdio: 'ignore',
  })
  child.unref()

  return {
    ok: true,
    status: 'starting',
    force: force === '1',
    message: force === '1' ? 'API reload scheduled' : 'API start scheduled',
  }
})
