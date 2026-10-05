// How move categories look: name, symbol and colour. The colours follow
// the chess analyzer next door (E:/Projects/chess_analyzer), so the two
// apps read the same way; "forced" is new here (checkers captures are
// compulsory, so many turns have only one legal move).
import type { Quality, Review, ReviewTurn } from './api'

export const QUALITY: Record<
  Quality,
  { label: string; symbol: string; color: string }
> = {
  great: { label: 'Great', symbol: '!', color: '#5c8bb0' },
  best: { label: 'Best', symbol: '★', color: '#769656' },
  excellent: { label: 'Excellent', symbol: '✓', color: '#8ba9c5' },
  good: { label: 'Good', symbol: '+', color: '#96af8c' },
  forced: { label: 'Forced', symbol: '□', color: '#9a9a9a' },
  inaccuracy: { label: 'Inaccuracy', symbol: '?!', color: '#f0c15f' },
  mistake: { label: 'Mistake', symbol: '?', color: '#e07b39' },
  miss: { label: 'Miss', symbol: '✕', color: '#ca3431' },
  blunder: { label: 'Blunder', symbol: '??', color: '#ca3431' },
}

/** Display order, best to worst. */
export const QUALITY_ORDER = Object.keys(QUALITY) as Quality[]

/** Categories marked on the graph: standout moves and errors (routine
 * ones would only clutter it). */
export const NOTABLE = new Set<Quality>(['great', 'mistake', 'miss', 'blunder'])

/** The review's verdicts that still apply to `actions` (the current line,
 * undone turns included): those whose turn was played the same way. */
export function validTurns(
  review: Review,
  reviewed: number[],
  actions: number[],
): ReviewTurn[] {
  let same = 0
  while (
    same < reviewed.length &&
    same < actions.length &&
    reviewed[same] === actions[same]
  ) {
    same++
  }
  return review.turns.filter((t) => t.ply + t.hops <= same)
}
