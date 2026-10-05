// The analysis table: every legal hop with the network's prior, the
// search's visits and Q, and the line the search expects after it.
// Hovering a row shows that line on the board; clicking plays the move.
import type { Analysis, MoveStats } from '../api'
import { lineText, percent, signed, turnText } from '../format'

interface Props {
  analysis: Analysis
  onHover: (move: MoveStats | null) => void
  onPlay: ((action: number) => void) | null
}

/** Moves sorted by visits, then prior: the search's ranking. */
export function ranked(moves: MoveStats[]): MoveStats[] {
  return [...moves].sort((a, b) => b.visits - a.visits || b.prior - a.prior)
}

/** The table. */
export function MoveTable({ analysis, onHover, onPlay }: Props) {
  const total = analysis.moves.reduce((sum, m) => sum + m.visits, 0)
  return (
    <table className="moves" onMouseLeave={() => onHover(null)}>
      <thead>
        <tr>
          <th>move</th>
          <th title="Network's prior probability">prior</th>
          <th title="Share of the search's visits">visits</th>
          <th title="Average value for the side to move (-1 to +1)">Q</th>
          <th>expected line</th>
        </tr>
      </thead>
      <tbody>
        {ranked(analysis.moves).map((m) => (
          <tr
            key={m.action}
            className={m.action === analysis.best ? 'best' : ''}
            onMouseEnter={() => onHover(m)}
            onClick={() => onPlay?.(m.action)}
            style={{ cursor: onPlay ? 'pointer' : 'default' }}
          >
            <td className="move">{turnText([m])}</td>
            <td>
              <Meter share={m.prior} />
            </td>
            <td>
              <Meter share={total ? m.visits / total : 0} />
              <span className="count">{m.visits}</span>
            </td>
            <td className="num">{m.q === null ? '' : signed(m.q)}</td>
            <td className="line">{lineText(m.line.slice(1))}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** A share as a small bar with its percentage. */
function Meter({ share }: { share: number }) {
  return (
    <span className="meter">
      <span className="meter-fill" style={{ width: `${share * 100}%` }} />
      <span className="meter-text">{percent(share)}</span>
    </span>
  )
}
