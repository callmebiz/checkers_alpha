// Types and calls for the Python server (see server/app.py and
// server/engine.py, which document every field). Squares are PDN numbers,
// 1-32; an action is one hop.

export type Color = 'black' | 'white'

/** A starting FEN (null: the standard start) and the hops played since. */
export interface Line {
  start: string | null
  actions: number[]
}

/** One hop, as drawn on the board. */
export interface Hop {
  action: number
  from: number
  to: number
  captures: number | null
}

export interface Piece {
  square: number
  color: Color
  king: boolean
}

/** Consecutive hops by one side; `ply` indexes the first in the line. */
export interface Turn {
  color: Color
  ply: number
  actions: number[]
  text: string
}

export interface GameState {
  fen: string | null
  turn: Color
  jump_square: number | null
  pieces: Piece[]
  legal: Hop[]
  result: { winner: Color | null; reason: string } | null
  turns: Turn[]
  last: Hop | null
  no_progress: number
  repetitions: number
}

/** A hop in a principal variation, with the side that plays it. */
export interface LineHop extends Hop {
  color: Color
}

/** A root move: the network's prior, the search's visits and Q (for the
 * side to move, -1 to 1), and the line expected after it. */
export interface MoveStats extends Hop {
  prior: number
  visits: number
  q: number | null
  line: LineHop[]
}

export interface Analysis {
  turn: Color
  sims: number
  net: { wdl: [number, number, number] | null; value: number } | null
  q: number | null
  /** The search's win/draw/loss (the network's, averaged over the tree):
   * unlike the network alone, it sees through exchanges in progress. */
  wdl: [number, number, number] | null
  moves: MoveStats[]
  best: number | null
}

export interface Checkpoint {
  model: string
  iteration: number
  hours: number | null
}

export interface Run {
  run: string
  shape: string
  checkpoints: Checkpoint[]
}

export interface Players {
  runs: Run[]
  engines: string[]
  default_model: string | null
  strongest_model: string | null
  device: string
}

/** POST JSON to the server; throws with the server's reason on failure. */
async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) throw new Error(await reason(response))
  return response.json() as Promise<T>
}

/** The error message in a failed response. */
async function reason(response: Response): Promise<string> {
  try {
    const data = await response.json()
    return String(data.detail ?? response.statusText)
  } catch {
    return response.statusText
  }
}

/** Models (by run) and plain engines the server can play with. */
export async function getPlayers(): Promise<Players> {
  const response = await fetch('/api/players')
  if (!response.ok) throw new Error(await reason(response))
  return response.json() as Promise<Players>
}

/** The position at the end of a line. */
export function getState(line: Line): Promise<GameState> {
  return post('/api/state', line)
}

/** The hop a player picks: `model:<name>:<sims>`, `ab:<depth>`, `random`. */
export function getMove(
  line: Line,
  player: string,
): Promise<{ move: Hop; analysis: Analysis | null }> {
  return post('/api/move', { line, player })
}

/** Where the live-analysis WebSocket is, relative to this page. */
export function liveUrl(): string {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws'
  return `${scheme}://${location.host}/api/live`
}

/** White's chances at one point of the game (see /api/evals): shares of
 * White winning, a draw and Black winning, and the search value for White
 * (-1 to 1) when the request searched. `turn` -1 is the start. */
export interface EvalPoint {
  turn: number
  ply: number
  white: number
  draw: number
  black: number
  q: number | null
}

/** The evaluation graph of a whole line. */
export async function getEvals(
  line: Line,
  model: string,
  sims?: number, // default: the server's (a light search per turn)
): Promise<EvalPoint[]> {
  const data = await post<{ points: EvalPoint[] }>('/api/evals', {
    line,
    model,
    sims,
  })
  return data.points
}

/** A move category from a review (see server/review.py). */
export type Quality =
  | 'forced'
  | 'great'
  | 'best'
  | 'excellent'
  | 'good'
  | 'inaccuracy'
  | 'mistake'
  | 'miss'
  | 'blunder'

/** The verdict on one turn: what it cost (share of a point given away),
 * the mover's expected score before and after, and the judge's choice. */
export interface ReviewTurn {
  color: Color
  ply: number
  hops: number
  text: string
  quality: Quality
  cost: number
  before: number
  after: number
  accuracy: number | null
  best: string | null
}

export interface Review {
  judge: string
  sims: number
  turns: ReviewTurn[]
  sides: Record<
    Color,
    { accuracy: number | null; counts: Record<Quality, number> }
  >
}

/** Judge every turn of a line (by the strongest checkpoint by default). */
export function getReview(line: Line, model?: string): Promise<Review> {
  return post('/api/review', { line, model })
}
