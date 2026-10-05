// Action decoding and move text agree with core/game.py.
import { describe, expect, it } from 'vitest'
import { decode, lineText, percent, turnText } from './format'

describe('decode', () => {
  it('reads steps and jumps', () => {
    // 11-15: square 11 (index 10) moving SW (2), no jump
    expect(decode(10 * 8 + 2 * 2)).toEqual({
      action: 84,
      from: 11,
      to: 15,
      captures: null,
    })
    // 22x15 jumping 18: square 22 (index 21) moving NE (1), jump
    expect(decode(21 * 8 + 1 * 2 + 1)).toMatchObject({
      from: 22,
      to: 15,
      captures: 18,
    })
  })
})

describe('move text', () => {
  it('writes steps and multi-jumps like PDN', () => {
    expect(turnText([decode(84)])).toBe('11-15')
    const a = { from: 22, to: 15, captures: 18, action: 0 }
    const b = { from: 15, to: 6, captures: 10, action: 0 }
    expect(turnText([a, b])).toBe('22x15x6')
  })

  it('groups a line into turns by colour', () => {
    const hops = [
      { ...decode(84), color: 'black' as const },
      { from: 22, to: 15, captures: 18, action: 0, color: 'white' as const },
      { from: 15, to: 6, captures: 10, action: 0, color: 'white' as const },
    ]
    expect(lineText(hops)).toBe('11-15 22x15x6')
  })

  it('formats shares', () => {
    expect(percent(0.423)).toBe('42%')
    expect(percent(0.035)).toBe('3.5%')
    expect(percent(0.001)).toBe('<1%')
  })
})
