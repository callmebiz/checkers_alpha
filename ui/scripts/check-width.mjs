// Fails if any source line is wider than 80 columns (the project's hard
// limit; Prettier wraps code but not long strings or comments).
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'

const LIMIT = 80
const EXTENSIONS = ['.ts', '.tsx', '.mjs', '.css', '.html']

/** Every source file under `dir`, recursively. */
function files(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name)
    if (statSync(path).isDirectory()) return files(path)
    return EXTENSIONS.some((e) => name.endsWith(e)) ? [path] : []
  })
}

const wide = ['src', 'scripts', 'index.html', 'vite.config.ts']
  .flatMap((p) => (statSync(p).isDirectory() ? files(p) : [p]))
  .flatMap((path) =>
    readFileSync(path, 'utf8')
      .split('\n')
      .map((line, i) => ({ path, line: i + 1, width: line.length }))
      .filter((l) => l.width > LIMIT),
  )
for (const l of wide) console.log(`${l.path}:${l.line}: ${l.width} columns`)
process.exit(wide.length ? 1 : 0)
