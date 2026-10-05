// Board geometry agrees with core/game.py's numbering.
import { describe, expect, it } from 'vitest'
import { rowColToSquare, squareCenter, squareToRowCol } from './geometry'

describe('squares', () => {
  it('number the dark squares like PDN', () => {
    expect(squareToRowCol(1)).toEqual([0, 1])
    expect(squareToRowCol(5)).toEqual([1, 0])
    expect(squareToRowCol(32)).toEqual([7, 6])
  })

  it('round-trip through rows and columns', () => {
    for (let s = 1; s <= 32; s++) {
      const [row, col] = squareToRowCol(s)
      expect((row + col) % 2).toBe(1)
      expect(rowColToSquare(row, col)).toBe(s)
    }
    expect(rowColToSquare(0, 0)).toBeNull()
  })

  it('rotate 180 degrees when flipped', () => {
    expect(squareCenter(1, true)).toEqual(squareCenter(32, false))
  })
})
