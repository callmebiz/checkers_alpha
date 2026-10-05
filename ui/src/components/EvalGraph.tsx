// The evaluation graph: White's, the draw's and Black's shares over the
// game as stacked areas (Black on top, White filling from the bottom, as
// in chess apps; checkers adds the draw band between them). The current
// turn is marked; hovering shows a turn's numbers and clicking goes there.
// Turns after the current one (undone, kept for redo) are dimmed.
import { useState } from 'react'
import type { EvalPoint, Quality } from '../api'
import { percent } from '../format'
import { NOTABLE, QUALITY } from '../quality'

const W = 1000 // drawing units; the SVG stretches to the panel's width
const H = 160

/** SVG path of an area between two curves (both in 0-1, 0 at the top). */
export function bandPath(top: number[], bottom: number[]): string {
  const n = top.length
  const x = (i: number) => (n > 1 ? (i / (n - 1)) * W : W / 2).toFixed(1)
  const y = (v: number) => (v * H).toFixed(1)
  const upper = top.map((v, i) => `${x(i)},${y(v)}`)
  const lower = bottom.map((v, i) => `${x(i)},${y(v)}`).reverse()
  return `M${upper.join('L')}L${lower.join('L')}Z`
}

/** The graph; `current` is the index of the current turn (-1: start). */
export function EvalGraph(props: {
  points: EvalPoint[]
  current: number
  labels: string[] // text of each turn, for the tooltip
  qualities: (Quality | undefined)[] // review verdicts by turn
  onGoto: (turn: number) => void
}) {
  const { points } = props
  const [hover, setHover] = useState<number | null>(null)
  if (points.length < 2) return null
  const n = points.length
  // boundaries from the top: Black's band, then the draw, then White's
  const blackEnd = points.map((p) => p.black)
  const drawEnd = points.map((p) => p.black + p.draw)
  const zeros = points.map(() => 0)
  const ones = points.map(() => 1)
  const xOf = (index: number) => (index / (n - 1)) * 100 // percent
  const at = (clientX: number, box: DOMRect) =>
    Math.round(((clientX - box.left) / box.width) * (n - 1))
  // the graph can lag a move behind while new evaluations load
  const currentX = xOf(Math.min(props.current + 1, n - 1))
  const shown = hover === null ? null : points[hover]

  return (
    <div className="graph">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        onMouseMove={(e) =>
          setHover(at(e.clientX, e.currentTarget.getBoundingClientRect()))
        }
        onMouseLeave={() => setHover(null)}
        onClick={(e) =>
          props.onGoto(
            at(e.clientX, e.currentTarget.getBoundingClientRect()) - 1,
          )
        }
      >
        <path d={bandPath(zeros, blackEnd)} className="g-black" />
        <path d={bandPath(blackEnd, drawEnd)} className="g-draw" />
        <path d={bandPath(drawEnd, ones)} className="g-white" />
        <line x1={0} x2={W} y1={H / 2} y2={H / 2} className="g-mid" />
        {/* dim the turns after the current one */}
        <rect
          x={(currentX / 100) * W}
          y={0}
          width={W - (currentX / 100) * W}
          height={H}
          className="g-future"
        />
      </svg>
      <div className="g-cursor" style={{ left: `${currentX}%` }} />
      {props.qualities.map((q, turn) => {
        const p = points[turn + 1]
        if (!q || !p || !NOTABLE.has(q)) return null
        // on White's expected-score line: the middle of the draw band
        const y = (p.black + p.draw / 2) * 100
        return (
          <div
            key={turn}
            className="g-dot"
            title={`${props.labels[turn] ?? ''}: ${QUALITY[q].label}`}
            style={{
              left: `${xOf(turn + 1)}%`,
              top: `${y}%`,
              background: QUALITY[q].color,
            }}
          />
        )
      })}
      {shown && hover !== null && (
        <>
          <div className="g-hover" style={{ left: `${xOf(hover)}%` }} />
          <div
            className="g-tip"
            style={{
              left: `${Math.min(Math.max(xOf(hover), 12), 88)}%`,
            }}
          >
            <b>{hover === 0 ? 'start' : (props.labels[hover - 1] ?? '')}</b>
            {` Black ${percent(shown.black)} · draw ${percent(shown.draw)}`}
            {` · White ${percent(shown.white)}`}
          </div>
        </>
      )}
    </div>
  )
}
