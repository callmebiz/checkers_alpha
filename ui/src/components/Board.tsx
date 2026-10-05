// The board: squares, pieces, the last move, click-to-move and analysis
// arrows. Click a piece that can move, then one of the marked squares; a
// multi-jump continues with the same piece, which stays selected. In edit
// mode it shows the position being set up, and clicks go to the editor.
//
// Arrows show the analysis: one per root move, thicker and more opaque the
// larger its share (of the search's visits, or of the network's prior).
// Hovering a move elsewhere passes its line here, drawn as numbered arrows.
import { useEffect, useState } from 'react'
import type {
  Color,
  GameState,
  LineHop,
  MoveStats,
  Piece,
  Quality,
} from '../api'
import { percent } from '../format'
import { QUALITY } from '../quality'
import { CELL, arrowEnds, squareCenter } from '../geometry'

export type ArrowMode = 'search' | 'prior'

interface Props {
  state: GameState
  flipped: boolean
  /** Whether the person at the board may move now. */
  canMove: boolean
  onPlay: (action: number) => void
  moves: MoveStats[] | null
  mode: ArrowMode
  best: number | null
  /** A line to draw instead of the move arrows (a hovered move's). */
  line: LineHop[] | null
  /** The review's verdict on the last turn, shown where it ended. */
  lastQuality: Quality | null
  /** Edit mode: the pieces being placed, and where clicks go. */
  edit: { pieces: Piece[]; onSquare: (square: number) => void } | null
}

const SIZE = 8 * CELL
const MIN_SHARE = 0.02 // moves below this share get no arrow
const LINE_HOPS = 8 // hops of a hovered line drawn (more is clutter)

/** Each move's share of the visits (or, before any search, the prior). */
export function shares(moves: MoveStats[], mode: ArrowMode): number[] {
  const total = moves.reduce((sum, m) => sum + m.visits, 0)
  if (mode === 'prior' || total === 0) return moves.map((m) => m.prior)
  return moves.map((m) => m.visits / total)
}

/** The SVG board. */
export function Board(props: Props) {
  const { state, flipped, canMove, onPlay } = props
  const [selected, setSelected] = useState<number | null>(null)

  // a new position clears the selection, except mid multi-jump
  useEffect(() => setSelected(state.jump_square), [state])

  const from = new Set(state.legal.map((h) => h.from))
  const targets = state.legal.filter((h) => h.from === selected)

  const editing = props.edit
  const pieces = editing ? editing.pieces : state.pieces

  /** Select a movable piece, or play to a marked square (in edit mode,
   * hand the square to the editor). */
  function click(square: number) {
    if (editing) return editing.onSquare(square)
    if (!canMove) return
    const hop = targets.find((h) => h.to === square)
    if (hop) onPlay(hop.action)
    else if (from.has(square) && state.jump_square === null) {
      setSelected(square === selected ? null : square)
    }
  }

  return (
    <svg
      className="board"
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      role="img"
      aria-label="Checkers board"
    >
      {Array.from({ length: 64 }, (_, i) => {
        const row = i >> 3
        const col = i & 7
        const dark = (row + col) % 2 === 1
        return (
          <rect
            key={i}
            x={col * CELL}
            y={row * CELL}
            width={CELL}
            height={CELL}
            className={dark ? 'sq-dark' : 'sq-light'}
          />
        )
      })}
      {!editing && state.last && (
        <>
          <Tint square={state.last.from} flipped={flipped} kind="last" />
          <Tint square={state.last.to} flipped={flipped} kind="last" />
        </>
      )}
      {!editing && selected !== null && (
        <Tint square={selected} flipped={flipped} kind="selected" />
      )}
      <Numbers flipped={flipped} />
      {pieces.map((p) => (
        <PieceShape
          key={p.square}
          {...p}
          flipped={flipped}
          movable={!editing && canMove && from.has(p.square)}
        />
      ))}
      {!editing && state.last && props.lastQuality && (
        <QualityMark
          square={state.last.to}
          flipped={flipped}
          quality={props.lastQuality}
        />
      )}
      {!editing &&
        targets.map((h) => {
          const [x, y] = squareCenter(h.to, flipped)
          return <circle key={h.to} cx={x} cy={y} r={14} className="target" />
        })}
      {editing ? null : props.line ? (
        <LineArrows line={props.line} flipped={flipped} />
      ) : (
        props.moves && (
          <MoveArrows
            moves={props.moves}
            mode={props.mode}
            best={props.best}
            flipped={flipped}
          />
        )
      )}
      {/* clicks land on transparent squares above everything */}
      {Array.from({ length: 32 }, (_, i) => {
        const [x, y] = squareCenter(i + 1, flipped)
        return (
          <rect
            key={i}
            x={x - CELL / 2}
            y={y - CELL / 2}
            width={CELL}
            height={CELL}
            className={editing || canMove ? 'hit can-move' : 'hit'}
            onClick={() => click(i + 1)}
          />
        )
      })}
    </svg>
  )
}

/** A coloured overlay on one square. */
function Tint(props: { square: number; flipped: boolean; kind: string }) {
  const [x, y] = squareCenter(props.square, props.flipped)
  return (
    <rect
      x={x - CELL / 2}
      y={y - CELL / 2}
      width={CELL}
      height={CELL}
      className={`tint-${props.kind}`}
    />
  )
}

/** Faint PDN numbers in the corner of each dark square, like the
 * coordinates on a chess board. */
function Numbers({ flipped }: { flipped: boolean }) {
  return (
    <>
      {Array.from({ length: 32 }, (_, i) => {
        const [x, y] = squareCenter(i + 1, flipped)
        return (
          <text key={i} x={x - 45} y={y - 31} className="sq-number">
            {i + 1}
          </text>
        )
      })}
    </>
  )
}

/** A man or king; kings carry a crown. */
function PieceShape(props: {
  square: number
  color: Color
  king: boolean
  flipped: boolean
  movable: boolean
}) {
  const [x, y] = squareCenter(props.square, props.flipped)
  const cls = `piece piece-${props.color}${props.movable ? ' movable' : ''}`
  return (
    <g className={cls}>
      <circle cx={x} cy={y + 4} r={38} className="piece-shadow" />
      <circle cx={x} cy={y} r={38} className="piece-body" />
      <circle cx={x} cy={y} r={27} className="piece-ring" />
      {props.king && (
        <path
          className="crown"
          d={
            `M ${x - 18} ${y + 10} L ${x - 20} ${y - 10} L ${x - 9} ${y} ` +
            `L ${x} ${y - 16} L ${x + 9} ${y} L ${x + 20} ${y - 10} ` +
            `L ${x + 18} ${y + 10} Z`
          }
        />
      )}
    </g>
  )
}

/** A verdict's symbol in a coloured circle at a square's corner. */
function QualityMark(props: {
  square: number
  flipped: boolean
  quality: Quality
}) {
  const [x, y] = squareCenter(props.square, props.flipped)
  const q = QUALITY[props.quality]
  return (
    <g className="mark">
      <circle cx={x + 34} cy={y - 34} r={17} fill={q.color} />
      <text x={x + 34} y={y - 27}>
        {q.symbol}
      </text>
    </g>
  )
}

/** Points of an arrow (shaft and head) as an SVG polygon. */
function arrowPoints(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  width: number,
): string {
  const length = Math.hypot(x2 - x1, y2 - y1)
  const ux = (x2 - x1) / length
  const uy = (y2 - y1) / length
  const head = Math.min(Math.max(width * 2.2, 22), length * 0.6)
  const half = Math.max(width * 1.25, 13)
  const bx = x2 - ux * head // where the head starts
  const by = y2 - uy * head
  const [px, py] = [-uy, ux] // perpendicular
  const w = width / 2
  const points = [
    [x1 + px * w, y1 + py * w],
    [bx + px * w, by + py * w],
    [bx + px * half, by + py * half],
    [x2, y2],
    [bx - px * half, by - py * half],
    [bx - px * w, by - py * w],
    [x1 - px * w, y1 - py * w],
  ]
  return points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ')
}

/** One arrow per move with a large enough share, labelled with it. */
function MoveArrows(props: {
  moves: MoveStats[]
  mode: ArrowMode
  best: number | null
  flipped: boolean
}) {
  const share = shares(props.moves, props.mode)
  // draw small arrows first so the important ones end up on top
  const order = props.moves
    .map((m, i) => ({ m, s: share[i] }))
    .filter((a) => a.s >= MIN_SHARE)
    .sort((a, b) => a.s - b.s)
  return (
    <g className="arrows">
      {order.map(({ m, s }) => {
        const [x1, y1, x2, y2] = arrowEnds(m.from, m.to, props.flipped)
        const best = m.action === props.best
        const width = 6 + 22 * Math.sqrt(s)
        return (
          <g key={m.action} opacity={0.35 + 0.6 * Math.sqrt(s)}>
            <polygon
              points={arrowPoints(x1, y1, x2, y2, width)}
              className={best ? 'arrow arrow-best' : 'arrow'}
            />
            {s >= 0.05 && (
              <text
                x={(x1 + x2) / 2}
                y={(y1 + y2) / 2 + 7}
                className="arrow-label"
              >
                {percent(s)}
              </text>
            )}
          </g>
        )
      })}
    </g>
  )
}

/** A principal variation: numbered arrows, coloured by the side moving. */
function LineArrows(props: { line: LineHop[]; flipped: boolean }) {
  return (
    <g className="arrows">
      {props.line.slice(0, LINE_HOPS).map((h, i) => {
        const [x1, y1, x2, y2] = arrowEnds(h.from, h.to, props.flipped)
        const fade = Math.max(0.25, 1 - i * 0.09)
        return (
          <g key={i} opacity={fade}>
            <polygon
              points={arrowPoints(x1, y1, x2, y2, 10)}
              className={`arrow line-${h.color}`}
            />
            <circle cx={x1} cy={y1} r={15} className={`step step-${h.color}`} />
            <text x={x1} y={y1 + 6} className={`step-text step-${h.color}`}>
              {i + 1}
            </text>
          </g>
        )
      })}
    </g>
  )
}
