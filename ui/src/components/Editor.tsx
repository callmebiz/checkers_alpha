// The position editor: pick a piece (or the eraser), click squares on the
// board, choose who moves, then load it as a new game. Loaded positions
// start paused, so the analysis can be studied before anyone moves.
import { useState } from 'react'
import { type Color, type Piece, getState } from '../api'

export type Tool = 'black' | 'black-king' | 'white' | 'white-king' | 'erase'

const TOOLS: { tool: Tool; label: string }[] = [
  { tool: 'black', label: 'Black man' },
  { tool: 'black-king', label: 'Black king' },
  { tool: 'white', label: 'White man' },
  { tool: 'white-king', label: 'White king' },
  { tool: 'erase', label: 'Eraser' },
]

/** The pieces of the standard start. */
export function startPieces(): Piece[] {
  return Array.from({ length: 24 }, (_, i) => ({
    square: i < 12 ? i + 1 : i + 9,
    color: i < 12 ? 'black' : 'white',
    king: false,
  }))
}

/** `pieces` after using `tool` on `square`: the eraser clears it, the
 * same piece again removes it, anything else replaces it. A man placed on
 * the row where it would be crowned becomes a king. */
export function applyTool(
  pieces: Piece[],
  square: number,
  tool: Tool,
): Piece[] {
  const rest = pieces.filter((p) => p.square !== square)
  if (tool === 'erase') return rest
  const color: Color = tool.startsWith('black') ? 'black' : 'white'
  const crowning = color === 'black' ? square >= 29 : square <= 4
  const king = tool.endsWith('king') || crowning
  const old = pieces.find((p) => p.square === square)
  if (old && old.color === color && old.king === king) return rest
  return [...rest, { square, color, king }]
}

/** A position as PDN FEN, e.g. `W:W21,K30:B1,2`. */
export function toFen(pieces: Piece[], turn: Color): string {
  /** One side's squares in order, kings prefixed with K. */
  const side = (color: Color) =>
    pieces
      .filter((p) => p.color === color)
      .sort((a, b) => a.square - b.square)
      .map((p) => `${p.king ? 'K' : ''}${p.square}`)
      .join(',')
  return `${turn === 'white' ? 'W' : 'B'}:W${side('white')}:B${side('black')}`
}

/** The editor's panel (the board itself shows the pieces being edited). */
export function Editor(props: {
  pieces: Piece[]
  turn: Color
  tool: Tool
  onTool: (tool: Tool) => void
  onTurn: (turn: Color) => void
  onPieces: (pieces: Piece[]) => void
  onCancel: () => void
  onLoad: (fen: string) => void
}) {
  const [error, setError] = useState<string | null>(null)
  const fen = toFen(props.pieces, props.turn)

  /** Check the position with the server, then load it. */
  function load() {
    getState({ start: fen, actions: [] }).then(
      () => props.onLoad(fen),
      (e: Error) => setError(e.message),
    )
  }

  return (
    <section className="box editor">
      <div className="box-head">
        <b>Edit position</b>
        <span className="muted">click squares to place pieces</span>
      </div>
      <div className="tools">
        {TOOLS.map(({ tool, label }) => (
          <button
            key={tool}
            className={`tool${props.tool === tool ? ' active' : ''}`}
            onClick={() => props.onTool(tool)}
            title={label}
          >
            <ToolIcon tool={tool} /> {label}
          </button>
        ))}
      </div>
      <div className="buttons">
        <label>
          To move:{' '}
          <select
            value={props.turn}
            onChange={(e) => props.onTurn(e.target.value as Color)}
          >
            <option value="black">Black</option>
            <option value="white">White</option>
          </select>
        </label>
        <button onClick={() => props.onPieces(startPieces())}>Start</button>
        <button onClick={() => props.onPieces([])}>Clear</button>
      </div>
      <code className="fen-preview">{fen}</code>
      {error && <p className="error">{error}</p>}
      <div className="buttons">
        <button className="primary" onClick={load}>
          Load position
        </button>
        <button onClick={props.onCancel}>Cancel</button>
      </div>
    </section>
  )
}

/** A small picture of a tool: a piece (crowned for kings) or an eraser. */
function ToolIcon({ tool }: { tool: Tool }) {
  if (tool === 'erase') {
    return (
      <svg viewBox="0 0 20 20" className="tool-icon">
        <path d="M5 5l10 10M15 5L5 15" className="erase" />
      </svg>
    )
  }
  return (
    <svg viewBox="0 0 20 20" className="tool-icon">
      <circle
        cx={10}
        cy={10}
        r={8}
        className={tool.startsWith('black') ? 'ti-black' : 'ti-white'}
      />
      {tool.endsWith('king') && (
        <path d="M6 12l-1-5 3 2.5 2-4 2 4 3-2.5-1 5z" className="ti-crown" />
      )}
    </svg>
  )
}
