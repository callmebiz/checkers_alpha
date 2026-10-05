// The play and analysis page: board and evaluation bar on the left; on the
// right, who plays each side, the live analysis, the game record and the
// position (FEN).
import { useEffect, useMemo, useState } from 'react'
import {
  type Color,
  type EvalPoint,
  type Line,
  type MoveStats,
  type Piece,
  type Players,
  type Review,
  getEvals,
  getPlayers,
  getReview,
} from './api'
import { type ArrowMode, Board } from './components/Board'
import { Controls } from './components/Controls'
import { EvalBar, byColor } from './components/EvalBar'
import { EvalGraph } from './components/EvalGraph'
import { MoveList, entries } from './components/MoveList'
import { MoveTable } from './components/MoveTable'
import { ModelSelect, PlayerSelect } from './components/PlayerSelect'
import { Editor, type Tool } from './components/Editor'
import { ReviewPanel } from './components/ReviewPanel'
import { percent, signed } from './format'
import { type Game, type Player, useGame } from './game'
import { useLiveAnalysis } from './live'
import { validTurns } from './quality'
import { applyTool } from './components/Editor'

/** Loads the list of models, then shows the page. */
export function App() {
  const [catalog, setCatalog] = useState<Players | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    getPlayers().then(setCatalog, (e: Error) => setError(e.message))
  }, [])
  if (error) {
    return (
      <p className="notice">
        Can't reach the server ({error}). Start it with{' '}
        <code>python serve.py</code>.
      </p>
    )
  }
  if (!catalog) return <p className="notice">Connecting…</p>
  return <Page catalog={catalog} />
}

const ANALYSIS_SIMS = [400, 1600, 6400, 25600, 0]

/** The page itself, once the models are known. */
function Page({ catalog }: { catalog: Players }) {
  const model = catalog.default_model
  const game = useGame({
    black: { kind: 'human' },
    white: model
      ? { kind: 'model', model, sims: 800 }
      : { kind: 'engine', spec: 'ab:6' },
  })
  const { state, players } = game
  const [flipped, setFlipped] = useState(false)
  const [analyse, setAnalyse] = useState(model !== null)
  const [analysisModel, setAnalysisModel] = useState(model ?? '')
  const [budget, setBudget] = useState(6400)
  const [mode, setMode] = useState<ArrowMode>('search')
  const [hovered, setHovered] = useState<MoveStats | null>(null)

  const request = useMemo(
    () =>
      analyse && analysisModel && state && !state.result
        ? { line: game.line, model: analysisModel, sims: budget }
        : null,
    [analyse, analysisModel, budget, game.line, state],
  )
  const live = useLiveAnalysis(request)
  // only analyses of the position on the board are shown
  const analysis = game.fresh && request ? live.analysis : null
  useEffect(() => setHovered(null), [game.line])

  // the position editor, while open: pieces, side to move, current tool
  const [editing, setEditing] = useState<{
    pieces: Piece[]
    turn: Color
  } | null>(null)
  const [tool, setTool] = useState<Tool>('black')
  const canMove = game.canMove && !editing

  /** Change a side's player (any time; it plays from its next move) and
   * face the board toward a lone person. */
  function setPlayer(color: Color, player: Player) {
    const next = { ...players, [color]: player }
    game.choosePlayers(next)
    const people = (['black', 'white'] as Color[]).filter(
      (c) => next[c].kind === 'human',
    )
    if (people.length === 1) setFlipped(people[0] === 'black')
  }

  useKeys(game, () => setFlipped((f) => !f))

  // the whole line, undone turns included, for the graph
  const whole = useMemo(
    () => ({
      start: game.line.start,
      actions: [...game.line.actions, ...game.future.flat()],
    }),
    [game.line, game.future],
  )
  const points = useEvals(whole, analysisModel || null)
  const labels = state ? numbered(entries(state.turns, game.future)) : []
  const review = useReview(whole)

  if (!state) return <p className="notice">Loading the position…</p>
  return (
    <div className="layout">
      <section className="board-area">
        <div className="board-row">
          <EvalBar analysis={analysis} flipped={flipped} />
          <div className="board-wrap">
            <Board
              state={state}
              flipped={flipped}
              canMove={canMove}
              onPlay={game.play}
              moves={analysis?.moves ?? null}
              mode={mode}
              best={analysis?.best ?? null}
              line={hovered?.line ?? null}
              lastQuality={review.byTurn[(state.turns.length || 0) - 1] ?? null}
              edit={
                editing && {
                  pieces: editing.pieces,
                  onSquare: (sq) =>
                    setEditing({
                      ...editing,
                      pieces: applyTool(editing.pieces, sq, tool),
                    }),
                }
              }
            />
          </div>
        </div>
        <Controls
          game={game}
          onFlip={() => setFlipped(!flipped)}
          onEdit={() => {
            game.setRunning(false)
            setEditing({ pieces: state.pieces, turn: state.turn })
          }}
        />
        <EvalGraph
          points={points}
          current={state.turns.length - 1}
          labels={labels}
          qualities={review.byTurn}
          onGoto={game.goto}
        />
      </section>
      <aside className="panel">
        {editing ? (
          <Editor
            pieces={editing.pieces}
            turn={editing.turn}
            tool={tool}
            onTool={setTool}
            onTurn={(turn) => setEditing({ ...editing, turn })}
            onPieces={(pieces) => setEditing({ ...editing, pieces })}
            onCancel={() => setEditing(null)}
            onLoad={(fen) => {
              setEditing(null)
              game.reset(fen, true)
            }}
          />
        ) : (
          <Status game={game} />
        )}
        <div className="players">
          {(['black', 'white'] as Color[]).map((c) => (
            <PlayerSelect
              key={c}
              label={c === 'black' ? 'Black' : 'White'}
              catalog={catalog}
              value={players[c]}
              onChange={(p) => setPlayer(c, p)}
            />
          ))}
        </div>

        <section className="box">
          <div className="box-head">
            <label>
              <input
                type="checkbox"
                checked={analyse}
                disabled={!model}
                onChange={(e) => setAnalyse(e.target.checked)}
              />{' '}
              Analysis
            </label>
            {model && (
              <ModelSelect
                catalog={catalog}
                value={analysisModel}
                onChange={setAnalysisModel}
              />
            )}
            <select
              value={budget}
              onChange={(e) => setBudget(+e.target.value)}
              title="Simulations to search"
            >
              {ANALYSIS_SIMS.map((n) => (
                <option key={n} value={n}>
                  {n ? `${n} sims` : 'until stopped'}
                </option>
              ))}
            </select>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as ArrowMode)}
              title="What the arrows show"
            >
              <option value="search">arrows: search</option>
              <option value="prior">arrows: network</option>
            </select>
          </div>
          {live.error && <p className="error">{live.error}</p>}
          {analysis ? (
            <>
              <Summary analysis={analysis} done={live.done} />
              <MoveTable
                analysis={analysis}
                onHover={setHovered}
                onPlay={canMove ? game.play : null}
              />
            </>
          ) : (
            <p className="muted">
              {state.result
                ? 'Game over.'
                : analyse
                  ? 'Searching…'
                  : 'Analysis is off.'}
            </p>
          )}
        </section>

        <ReviewPanel
          review={review.review}
          turns={review.turns}
          current={review.turns[state.turns.length - 1] ?? null}
          busy={review.busy}
          error={review.error}
          canReview={whole.actions.length > 0}
          judgeLabel={modelLabel(catalog.strongest_model)}
          turnCount={labels.length}
          onReview={review.run}
        />
        <section className="box">
          <MoveList
            turns={state.turns}
            future={game.future}
            qualities={review.byTurn}
            onGoto={game.goto}
          />
        </section>
        <Position game={game} />
      </aside>
    </div>
  )
}

/** Whose move it is, who is thinking, or how the game ended. */
function Status({ game }: { game: Game }) {
  const { state, error } = game
  if (!state) return null
  const name = (c: Color) => (c === 'black' ? 'Black' : 'White')
  let text = `${name(state.turn)} to move`
  if (state.jump_square)
    text += ` (continue the jump from ${state.jump_square})`
  if (game.thinking) text = `${name(state.turn)} is thinking…`
  else if (!game.running && !state.result) {
    text += ' · paused: move either side'
  }
  if (state.result) {
    text = state.result.winner
      ? `${name(state.result.winner)} wins: ${state.result.reason}`
      : `Draw by ${state.result.reason}`
  }
  return (
    <div className={`status${state.result ? ' over' : ''}`}>
      <span className={`dot dot-${state.turn}`} /> {text}
      {error && <p className="error">{error}</p>}
    </div>
  )
}

/** The analysis in numbers: search size, network and search values. */
function Summary(props: {
  analysis: NonNullable<ReturnType<typeof useLiveAnalysis>['analysis']>
  done: boolean
}) {
  const { analysis } = props
  const wdl = byColor(analysis)
  const net = byColor(analysis, true)
  /** Black / draw / White shares as text. */
  const shares = (v: NonNullable<typeof wdl>) =>
    `Black ${percent(v.black)} · draw ${percent(v.draw)} · ` +
    `White ${percent(v.white)}`
  return (
    <p className="summary">
      <span>
        {analysis.sims.toLocaleString()} sims{props.done ? '' : '…'}
      </span>
      {wdl && (
        <span title="What the search concludes (the bar shows this)">
          search: {shares(wdl)} ({signed(analysis.q ?? 0)} for {analysis.turn})
        </span>
      )}
      {net && (
        <span title="The network's instant judgement, before searching">
          network alone: {shares(net)}
        </span>
      )}
    </p>
  )
}

/** The current FEN, copyable, and a box to load another. */
function Position({ game }: { game: Game }) {
  const [text, setText] = useState('')
  const fen = game.state?.fen
  return (
    <section className="box position">
      <div className="fen">
        <code>{fen ?? '(mid multi-jump: no FEN)'}</code>
        {fen && (
          <button onClick={() => navigator.clipboard.writeText(fen)}>
            Copy
          </button>
        )}
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault()
          if (text.trim()) game.reset(text.trim(), true)
        }}
      >
        <input
          value={text}
          placeholder="Paste a FEN, e.g. W:W21-32:B1-12"
          onChange={(e) => setText(e.target.value)}
        />
        <button type="submit">Load</button>
      </form>
    </section>
  )
}

/** A short name for a checkpoint: run and iteration. */
export function modelLabel(model: string | null): string {
  if (!model) return 'none'
  const parts = model.split('/')
  const run = parts.at(-3) ?? model
  const iteration = Number(parts.at(-1)?.match(/\d+/)?.[0] ?? NaN)
  return Number.isNaN(iteration) ? run : `${run} · iter ${iteration + 1}`
}

/** A review of the whole line on request; verdicts stay shown for turns
 * still played the way they were reviewed. */
function useReview(whole: Line) {
  const [done, setDone] = useState<{ review: Review; line: Line } | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const sameStart = done?.line.start === whole.start
  const turns =
    done && sameStart
      ? validTurns(done.review, done.line.actions, whole.actions)
      : []
  /** Ask the server for a review of the line as it is now. */
  const run = () => {
    const line = whole
    setBusy(true)
    setError(null)
    getReview(line)
      .then((review) => setDone({ review, line }))
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(false))
  }
  return {
    review: sameStart ? (done?.review ?? null) : null,
    turns,
    byTurn: turns.map((t) => t.quality),
    busy,
    error,
    run,
  }
}

/** Turn texts with move numbers, as in the record: `5. 11-15` for
 * Black, `5... 22-18` for White. */
export function numbered(list: { color: Color; text: string }[]): string[] {
  let move = 0
  return list.map((e, i) => {
    if (e.color === 'black' || i === 0) move++
    return `${move}${e.color === 'black' ? '.' : '...'} ${e.text}`
  })
}

/** Evaluations of every turn of `line` by `model` for the graph,
 * refreshed shortly after the line stops changing. */
function useEvals(line: Line, model: string | null): EvalPoint[] {
  const [points, setPoints] = useState<EvalPoint[]>([])
  const key = model ? JSON.stringify(line) : null
  useEffect(() => {
    if (!key || !model) return setPoints([])
    let cancelled = false
    const timer = setTimeout(() => {
      getEvals(JSON.parse(key), model).then(
        (p) => !cancelled && setPoints(p),
        () => !cancelled && setPoints([]),
      )
    }, 250)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [key, model])
  return points
}

/** Keyboard, as in chess apps: arrows step, Home and End jump, space
 * plays or pauses, F flips the board. */
function useKeys(game: Game, flip: () => void) {
  useEffect(() => {
    /** Handle one key press (ignored while typing in a field). */
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName)) return
      const keys: Record<string, () => void> = {
        ArrowLeft: game.back,
        ArrowRight: game.forward,
        Home: game.toStart,
        End: game.toEnd,
        ' ': () => game.setRunning(!game.running),
        f: flip,
      }
      if (!keys[e.key]) return
      e.preventDefault()
      keys[e.key]()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [game, flip])
}
