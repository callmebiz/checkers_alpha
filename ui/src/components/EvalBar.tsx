// The evaluation bar beside the board: the search's win/draw/loss as
// three stacked bands, Black's win at Black's end of the board, so it
// reads the same way as the board does. (Not the network alone: in the
// middle of an exchange it judges a side that just captured as winning,
// before the forced recapture.)
import type { Analysis } from '../api'
import { percent, signed } from '../format'

/** Win/draw/loss by colour from an analysis (values are for the side to
 * move): the search's, or the network's with `network`; null without. */
export function byColor(
  analysis: Analysis | null,
  network = false,
): { black: number; draw: number; white: number } | null {
  const wdl = network
    ? analysis?.net?.wdl
    : (analysis?.wdl ?? analysis?.net?.wdl)
  if (!analysis || !wdl) return null
  const [win, draw, loss] = wdl
  return analysis.turn === 'black'
    ? { black: win, draw, white: loss }
    : { black: loss, draw, white: win }
}

/** The bar; `flipped` puts Black at the bottom, as on the board. */
export function EvalBar(props: {
  analysis: Analysis | null
  flipped: boolean
}) {
  const v = byColor(props.analysis)
  const q = props.analysis?.q
  // search value for White (q is for the side to move)
  const white = q == null ? null : props.analysis?.turn === 'white' ? q : -q
  const bands = v
    ? [
        { name: 'black', share: v.black },
        { name: 'draw', share: v.draw },
        { name: 'white', share: v.white },
      ]
    : []
  if (props.flipped) bands.reverse()
  return (
    <div className="eval">
      <div className="eval-bar" aria-label="Search's win, draw and loss">
        {bands.map((b) => (
          <div
            key={b.name}
            className={`band band-${b.name}`}
            style={{ flexGrow: b.share }}
            title={`${b.name === 'draw' ? 'draw' : `${b.name} wins`}: ${percent(
              b.share,
            )}`}
          >
            {b.share >= 0.12 && percent(b.share)}
          </div>
        ))}
      </div>
      <div className="eval-q" title="Search value for White (-1 to +1)">
        {white == null ? '' : signed(white)}
      </div>
    </div>
  )
}
