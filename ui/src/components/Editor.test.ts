// The editor places pieces sensibly and writes FEN the engine reads.
import { describe, expect, it } from 'vitest'
import { applyTool, startPieces, toFen } from './Editor'

describe('applyTool', () => {
  it('places, replaces, toggles and erases', () => {
    let pieces = applyTool([], 14, 'black')
    expect(pieces).toEqual([{ square: 14, color: 'black', king: false }])
    pieces = applyTool(pieces, 14, 'white-king')
    expect(pieces).toEqual([{ square: 14, color: 'white', king: true }])
    expect(applyTool(pieces, 14, 'white-king')).toEqual([])
    expect(applyTool(pieces, 14, 'erase')).toEqual([])
  })

  it('crowns men placed on their crowning row', () => {
    expect(applyTool([], 30, 'black')[0].king).toBe(true)
    expect(applyTool([], 2, 'white')[0].king).toBe(true)
    expect(applyTool([], 2, 'black')[0].king).toBe(false)
  })
})

describe('toFen', () => {
  it('writes the standard start', () => {
    expect(toFen(startPieces(), 'black')).toBe(
      'B:W21,22,23,24,25,26,27,28,29,30,31,32:B1,2,3,4,5,6,7,8,9,10,11,12',
    )
  })

  it('marks kings and the side to move', () => {
    const pieces = [
      { square: 30, color: 'white' as const, king: true },
      { square: 1, color: 'black' as const, king: false },
    ]
    expect(toFen(pieces, 'white')).toBe('W:WK30:B1')
  })
})
