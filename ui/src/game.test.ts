// Redo, rewind and undo rules of the game hook.
import { describe, expect, it } from 'vitest'
import type { Turn } from './api'
import {
  autoplay,
  canMoveNow,
  playHop,
  playerSpec,
  rewind,
  undoDepth,
  type Players,
} from './game'

/** Turns of a line where each side played one hop, alternating. */
function turns(n: number): Turn[] {
  return Array.from({ length: n }, (_, i) => ({
    color: i % 2 ? 'white' : 'black',
    ply: i,
    actions: [100 + i],
    text: '',
  }))
}

const human = { kind: 'human' } as const
const engine = { kind: 'engine', spec: 'ab:4' } as const

describe('playHop', () => {
  it('keeps the redo list when the next undone hop is played', () => {
    expect(playHop([1], [[2, 3], [4]], 2)).toEqual({
      actions: [1, 2],
      future: [[3], [4]],
    })
    expect(playHop([1], [[2], [4]], 2).future).toEqual([[4]])
  })

  it('drops it for any other move', () => {
    expect(playHop([1], [[2], [4]], 9)).toEqual({
      actions: [1, 9],
      future: [],
    })
  })
})

describe('rewind', () => {
  it('moves later turns to the redo list', () => {
    const t = turns(3)
    expect(rewind([100, 101, 102], [[7]], t, 1)).toEqual({
      actions: [100],
      future: [[101], [102], [7]],
    })
  })
})

describe('undoDepth', () => {
  it('takes back two turns against a computer, one otherwise', () => {
    const vsEngine: Players = { black: human, white: engine }
    expect(undoDepth(turns(2), vsEngine)).toBe(2)
    expect(undoDepth(turns(1), vsEngine)).toBe(1)
    expect(undoDepth(turns(2), { black: human, white: human })).toBe(1)
    expect(undoDepth([], vsEngine)).toBe(0)
  })
})

describe('playerSpec', () => {
  it('names computer players for the server', () => {
    expect(playerSpec({ kind: 'model', model: 'a/b.pt', sims: 8 })).toBe(
      'model:a/b.pt:8',
    )
    expect(playerSpec(engine)).toBe('ab:4')
    expect(playerSpec(human)).toBeNull()
  })
})

describe('autoplay', () => {
  const base = {
    atEnd: true,
    over: false,
    personToMove: false,
    anyPerson: false,
    running: false,
    step: false,
  }

  it('lets computers move only while running, or one step', () => {
    expect(autoplay(base)).toBe('none')
    expect(autoplay({ ...base, running: true })).toBe('compute')
    expect(autoplay({ ...base, step: true })).toBe('compute')
  })

  it('answers a person while running, and waits while paused', () => {
    const vsPerson = { ...base, anyPerson: true }
    expect(autoplay({ ...vsPerson, running: true })).toBe('compute')
    expect(autoplay(vsPerson)).toBe('none')
    expect(autoplay({ ...vsPerson, personToMove: true, running: true })).toBe(
      'none',
    )
  })

  it('replays undone turns only in computer games', () => {
    const behind = { ...base, atEnd: false, running: true }
    expect(autoplay(behind)).toBe('redo')
    expect(autoplay({ ...behind, anyPerson: true })).toBe('none')
  })

  it('stops when the game is over', () => {
    expect(autoplay({ ...base, over: true, running: true })).toBe('none')
  })
})

describe('canMoveNow', () => {
  it("takes only a person's own moves while running", () => {
    expect(canMoveNow({ over: false, running: true, personToMove: true })).toBe(
      true,
    )
    expect(
      canMoveNow({ over: false, running: true, personToMove: false }),
    ).toBe(false)
  })

  it('takes moves for either side while paused, none once over', () => {
    expect(
      canMoveNow({ over: false, running: false, personToMove: false }),
    ).toBe(true)
    expect(canMoveNow({ over: true, running: false, personToMove: true })).toBe(
      false,
    )
  })
})
