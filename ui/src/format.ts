// Moves as text. Actions decode here too (as in core/game.py: square * 8 +
// direction * 2 + is_jump), so moves the server hasn't described, like the
// redo list, can still be shown.
import type { Color, Hop } from './api'
import { rowColToSquare, squareToRowCol } from './geometry'

// row and column steps of the directions NW, NE, SW, SE
const DIRECTIONS: [number, number][] = [
  [-1, -1],
  [-1, 1],
  [1, -1],
  [1, 1],
]

/** The hop an action makes. */
export function decode(action: number): Hop {
  const square = (action >> 3) + 1
  const [dr, dc] = DIRECTIONS[(action >> 1) & 3]
  const jump = action & 1
  const [row, col] = squareToRowCol(square)
  const step = jump ? 2 : 1
  return {
    action,
    from: square,
    to: rowColToSquare(row + dr * step, col + dc * step) ?? 0,
    captures: jump ? rowColToSquare(row + dr, col + dc) : null,
  }
}

/** PDN text of one turn's hops: `11-15`, or `22x15x6` for a multi-jump. */
export function turnText(hops: Hop[]): string {
  if (!hops.length) return ''
  if (hops[0].captures === null) return `${hops[0].from}-${hops[0].to}`
  return [hops[0].from, ...hops.map((h) => h.to)].join('x')
}

/** Group a line of hops into turns (consecutive hops by one side). */
export function groupTurns<T extends Hop & { color: Color }>(
  hops: T[],
): { color: Color; hops: T[] }[] {
  const out: { color: Color; hops: T[] }[] = []
  for (const h of hops) {
    const last = out[out.length - 1]
    if (last && last.color === h.color) last.hops.push(h)
    else out.push({ color: h.color, hops: [h] })
  }
  return out
}

/** A line of hops as PDN turns, e.g. `11-15 22-18 15x22`. */
export function lineText(hops: (Hop & { color: Color })[]): string {
  return groupTurns(hops)
    .map((t) => turnText(t.hops))
    .join(' ')
}

/** A share as a short percentage: `42%`, `3.5%`, `<1%`. */
export function percent(share: number): string {
  if (share >= 0.1) return `${Math.round(share * 100)}%`
  if (share >= 0.01) return `${(share * 100).toFixed(1)}%`
  return share > 0 ? '<1%' : '0%'
}

/** A value from -1 to 1 with its sign: `+0.12`. */
export function signed(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}`
}
