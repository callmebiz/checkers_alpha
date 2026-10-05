// The game record: turns numbered in pairs, Black then White, including
// undone turns (greyed) that redo would replay. Click a turn to go there.
import type { Color, Quality, Turn } from '../api'
import { decode, turnText } from '../format'
import { Badge } from './ReviewPanel'

interface Entry {
  color: Color
  text: string
  undone: boolean
}

/** Played turns, then undone ones (their colours alternate on). */
export function entries(turns: Turn[], future: number[][]): Entry[] {
  const out: Entry[] = turns.map((t) => ({
    color: t.color,
    text: t.text,
    undone: false,
  }))
  let color: Color = turns.at(-1)?.color === 'black' ? 'white' : 'black'
  for (const actions of future) {
    out.push({ color, text: turnText(actions.map(decode)), undone: true })
    color = color === 'black' ? 'white' : 'black'
  }
  return out
}

/** The record, as rows of (number, Black's turn, White's turn). */
export function MoveList(props: {
  turns: Turn[]
  future: number[][]
  /** Review verdicts by turn index (missing: not reviewed). */
  qualities: (Quality | undefined)[]
  onGoto: (index: number) => void
}) {
  const list = entries(props.turns, props.future)
  // pair turns into rows; a game starting with White leaves a gap
  const rows: (number | null)[][] = []
  list.forEach((e, i) => {
    if (e.color === 'black' || !rows.length) rows.push([null, null])
    rows[rows.length - 1][e.color === 'black' ? 0 : 1] = i
  })
  const current = props.turns.length - 1
  return (
    <div className="record">
      <button
        className={`turn${current === -1 ? ' current' : ''}`}
        onClick={() => props.onGoto(-1)}
      >
        start
      </button>
      <ol>
        {rows.map((row, r) => (
          <li key={r}>
            <span className="num">{r + 1}.</span>
            {row.map((i, side) =>
              i === null ? (
                <span key={side} className="turn empty">
                  …
                </span>
              ) : (
                <button
                  key={side}
                  className={
                    'turn' +
                    (list[i].undone ? ' undone' : '') +
                    (i === current ? ' current' : '')
                  }
                  onClick={() => props.onGoto(i)}
                >
                  {list[i].text}
                  {props.qualities[i] && <Badge quality={props.qualities[i]} />}
                </button>
              ),
            )}
          </li>
        ))}
      </ol>
    </div>
  )
}
