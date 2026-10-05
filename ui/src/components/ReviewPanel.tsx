// Game review: a button to have the strongest network judge every turn,
// then each side's accuracy and how many moves fell in each category, as
// in chess apps' game reports, plus the verdict on the current turn.
import type { Color, Review, ReviewTurn } from '../api'
import { QUALITY, QUALITY_ORDER } from '../quality'

interface Props {
  review: Review | null
  turns: ReviewTurn[] // verdicts still valid for the game on the board
  current: ReviewTurn | null // the verdict on the turn just played
  busy: boolean
  error: string | null
  canReview: boolean
  judgeLabel: string
  turnCount: number
  onReview: () => void
}

/** The panel. */
export function ReviewPanel(props: Props) {
  const { review, current } = props
  // ~70 ms per turn on an RTX 3070 at the review's search depth
  const estimate = Math.max(2, Math.round(props.turnCount * 0.07))
  return (
    <section className="box review">
      <div className="box-head">
        <b>Game review</b>
        <button
          onClick={props.onReview}
          disabled={!props.canReview || props.busy}
        >
          {props.busy
            ? `Reviewing… (~${estimate} s)`
            : review
              ? 'Review again'
              : 'Review game'}
        </button>
        <span className="muted" title="The best-rated checkpoint">
          judge: {props.judgeLabel}
        </span>
      </div>
      {props.error && <p className="error">{props.error}</p>}
      {current && <Verdict turn={current} />}
      {review && (
        <table className="tally">
          <thead>
            <tr>
              <th />
              <th>Black</th>
              <th />
              <th>White</th>
            </tr>
          </thead>
          <tbody>
            <tr className="accuracy">
              <td>Accuracy</td>
              <td className="count">
                <span className="acc acc-black">
                  {accuracyText(props.turns, 'black')}
                </span>
              </td>
              <td />
              <td className="count">
                <span className="acc acc-white">
                  {accuracyText(props.turns, 'white')}
                </span>
              </td>
            </tr>
            {QUALITY_ORDER.map((q) => (
              <tr key={q}>
                <td>{QUALITY[q].label}</td>
                <td className="count" style={{ color: QUALITY[q].color }}>
                  {count(props.turns, 'black', q)}
                </td>
                <td>
                  <Badge quality={q} />
                </td>
                <td className="count" style={{ color: QUALITY[q].color }}>
                  {count(props.turns, 'white', q)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}

/** How many of `side`'s turns fell in category `q`. */
function count(turns: ReviewTurn[], side: Color, q: string): number {
  return turns.filter((t) => t.color === side && t.quality === q).length
}

/** A side's accuracy over its unforced moves, or a dash. */
export function accuracyText(turns: ReviewTurn[], side: Color): string {
  const scored = turns
    .filter((t) => t.color === side && t.accuracy !== null)
    .map((t) => t.accuracy!)
  if (!scored.length) return '–'
  return (scored.reduce((a, b) => a + b, 0) / scored.length).toFixed(1)
}

/** A category's symbol in its colour. */
export function Badge(props: { quality: keyof typeof QUALITY }) {
  const q = QUALITY[props.quality]
  return (
    <span
      className="badge"
      style={{ background: q.color }}
      title={q.label}
      aria-label={q.label}
    >
      {q.symbol}
    </span>
  )
}

/** The verdict on one turn, in words. */
function Verdict({ turn }: { turn: ReviewTurn }) {
  const q = QUALITY[turn.quality]
  const lost = Math.round(turn.cost * 100)
  const alt =
    turn.best && turn.best !== turn.text && turn.quality !== 'forced'
      ? ` Best was ${turn.best}.`
      : ''
  return (
    <p className="verdict">
      <Badge quality={turn.quality} /> <b>{turn.text}</b> is{' '}
      {turn.quality === 'forced' ? 'forced' : q.label.toLowerCase()}
      {lost >= 1 && ` (gives away ${lost}% of a point)`}.{alt}{' '}
      <span className="muted">
        {turn.color === 'black' ? 'Black' : 'White'} expected{' '}
        {Math.round(turn.before * 100)}% → {Math.round(turn.after * 100)}%
      </span>
    </p>
  )
}
