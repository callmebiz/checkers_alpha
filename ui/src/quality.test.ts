// Review verdicts stay attached only to turns still played the same way.
import { describe, expect, it } from 'vitest'
import type { Review, ReviewTurn } from './api'
import { validTurns } from './quality'

/** A verdict on a turn of `hops` hops starting at `ply`. */
function turn(ply: number, hops = 1): ReviewTurn {
  return {
    color: ply % 2 ? 'white' : 'black',
    ply,
    hops,
    text: '',
    quality: 'best',
    cost: 0,
    before: 0.5,
    after: 0.5,
    accuracy: 100,
    best: null,
  }
}

const review = {
  turns: [turn(0), turn(1, 2), turn(3)],
} as Review

describe('validTurns', () => {
  it('keeps every verdict while the line is unchanged', () => {
    expect(validTurns(review, [1, 2, 3, 4], [1, 2, 3, 4])).toHaveLength(3)
  })

  it('drops verdicts from the first changed hop on', () => {
    // the second turn's second hop differs
    expect(validTurns(review, [1, 2, 3, 4], [1, 2, 9, 4])).toHaveLength(1)
    expect(validTurns(review, [1, 2, 3, 4], [1, 2, 3])).toHaveLength(2)
  })
})
