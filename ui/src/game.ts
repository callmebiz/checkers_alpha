// The game being played or explored: a line of hops plus a redo list,
// the server's view of its position, and who plays each side.
//
// Play and pause decide whether computers move on their own. While the
// game runs, a person moves only their own side; paused, any piece can be
// moved for either side (to force a move on a computer and see how it
// copes), and computers wait. Games with a person start running; games
// between computers start paused, step with forward, and replay undone
// turns while running. Players can be changed at any point.
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  type Color,
  type GameState,
  type Line,
  type Turn,
  getMove,
  getState,
} from './api'

/** Who plays a side: a person clicking, a network searching, or an engine
 * (alpha-beta or random) given by its spec. */
export type Player =
  | { kind: 'human' }
  | { kind: 'model'; model: string; sims: number }
  | { kind: 'engine'; spec: string }

export type Players = Record<Color, Player>

/** The server's player spec for a computer player, or null for a person. */
export function playerSpec(player: Player): string | null {
  if (player.kind === 'model') return `model:${player.model}:${player.sims}`
  if (player.kind === 'engine') return player.spec
  return null
}

/** Hops undone, grouped by turn, so redo can replay a whole turn. */
export type Future = number[][]

/** Play `action` after `actions`. If it's the next redo hop, the redo list
 * is kept (minus that hop); any other move starts a new branch. */
export function playHop(
  actions: number[],
  future: Future,
  action: number,
): { actions: number[]; future: Future } {
  const next = [...actions, action]
  if (!future.length || future[0][0] !== action) {
    return { actions: next, future: [] }
  }
  const rest = future[0].slice(1)
  const later = future.slice(1)
  return { actions: next, future: rest.length ? [rest, ...later] : later }
}

/** Go back to just before turn `index`, keeping the turns from there on
 * (and anything already undone) for redo. */
export function rewind(
  actions: number[],
  future: Future,
  turns: Turn[],
  index: number,
): { actions: number[]; future: Future } {
  if (index >= turns.length) return { actions, future }
  return {
    actions: actions.slice(0, turns[index].ply),
    future: [...turns.slice(index).map((t) => t.actions), ...future],
  }
}

/** How many turns undo takes back: one, or two when that would leave a
 * computer to move against a person (so the person gets their turn back). */
export function undoDepth(turns: Turn[], players: Players): number {
  if (!turns.length) return 0
  const last = turns[turns.length - 1]
  const human = (c: Color) => players[c].kind === 'human'
  const other: Color = last.color === 'black' ? 'white' : 'black'
  return !human(last.color) && human(other) && turns.length > 1 ? 2 : 1
}

/** Whether any side is played by a person. */
export function anyPerson(players: Players): boolean {
  return players.black.kind === 'human' || players.white.kind === 'human'
}

/** Whether the board takes a move now: while the game runs, only a
 * person's own move; while paused, a move for either side. */
export function canMoveNow(o: {
  over: boolean
  running: boolean
  personToMove: boolean
}): boolean {
  return !o.over && (!o.running || o.personToMove)
}

/** What should happen next on its own (see the file comment): replay an
 * undone turn, have a computer move, or nothing. */
export function autoplay(o: {
  atEnd: boolean // no undone turns left to replay
  over: boolean
  personToMove: boolean
  anyPerson: boolean
  running: boolean
  step: boolean // one computer move was asked for
}): 'redo' | 'compute' | 'none' {
  // computer games replay undone turns while playing; with a person,
  // stepping back must stay put
  if (!o.atEnd) return o.running && !o.anyPerson ? 'redo' : 'none'
  if (o.over || o.personToMove) return 'none'
  return o.running || o.step ? 'compute' : 'none'
}

/** Everything the board and panels need about the game, and the actions
 * that change it. */
export function useGame(initialPlayers: Players) {
  const [line, setLine] = useState<Line>({ start: null, actions: [] })
  const [future, setFuture] = useState<Future>([])
  // the latest state the server sent, and the line it belongs to
  const [loaded, setLoaded] = useState<{ line: Line; state: GameState }>()
  const [error, setError] = useState<string | null>(null)
  const [players, setPlayers] = useState<Players>(initialPlayers)
  // whether computers move on their own (games with a person start so)
  const [running, setRunning] = useState(anyPerson(initialPlayers))
  const [step, setStep] = useState(false)
  const [thinking, setThinking] = useState(false)
  // ms between moves while computers play each other, to follow the game
  const [delay, setDelay] = useState(500)
  const latest = useRef(line)

  // fetch the position whenever the line changes (latest request wins)
  useEffect(() => {
    latest.current = line
    getState(line).then(
      (state) => {
        if (latest.current !== line) return
        setLoaded({ line, state })
        setError(null)
      },
      (e: Error) => latest.current === line && setError(e.message),
    )
  }, [line])

  const state = loaded?.state ?? null
  // whether `state` is the position at the end of the current line
  const fresh = loaded?.line === line
  const person = (c: Color) => players[c].kind === 'human'
  const canMove =
    fresh &&
    !!state &&
    canMoveNow({
      over: !!state.result,
      running,
      personToMove: person(state.turn),
    })

  /** Play one hop at the end of the line. */
  const play = useCallback(
    (action: number) => {
      const next = playHop(line.actions, future, action)
      setLine({ start: line.start, actions: next.actions })
      setFuture(next.future)
    },
    [line, future],
  )

  /** Replay the next undone turn. */
  const redo = useCallback(() => {
    if (!future.length) return
    setLine({ start: line.start, actions: [...line.actions, ...future[0]] })
    setFuture(future.slice(1))
  }, [line, future])

  // replay or compute the next move when it's due
  useEffect(() => {
    if (!fresh || !state) return
    const isPerson = (c: Color) => players[c].kind === 'human'
    const people = anyPerson(players)
    const next = autoplay({
      atEnd: !future.length,
      over: !!state.result,
      personToMove: isPerson(state.turn),
      anyPerson: people,
      running,
      step,
    })
    if (next === 'none') {
      // a finished computer game stops; with a person it stays on
      if (running && state.result && !people) setRunning(false)
      if (step) setStep(false)
      return
    }
    if (next === 'redo') {
      const timer = setTimeout(redo, delay)
      return () => clearTimeout(timer)
    }
    let cancelled = false
    const spec = playerSpec(players[state.turn])!
    const wait = step || people ? 0 : delay
    setThinking(true)
    const timer = setTimeout(() => {
      getMove(line, spec)
        .then(({ move }) => {
          if (cancelled) return
          setStep(false)
          play(move.action)
        })
        .catch((e: Error) => {
          if (cancelled) return
          setError(e.message)
          setRunning(false)
        })
        .finally(() => !cancelled && setThinking(false))
    }, wait)
    return () => {
      cancelled = true
      clearTimeout(timer)
      setThinking(false)
    }
  }, [fresh, state, players, running, step, future, line, play, redo, delay])

  /** Change who plays; takes effect from the next move. A game left with
   * only computers pauses (press play); one gaining a person runs. */
  const choosePlayers = useCallback(
    (next: Players) => {
      if (!anyPerson(next)) setRunning(false)
      else if (!anyPerson(players)) setRunning(true)
      setPlayers(next)
    },
    [players],
  )

  /** Go to just after turn `index` of the whole line, undone turns
   * included (-1: the start); later turns stay available for redo.
   * A computer game pauses. */
  const goto = useCallback(
    (index: number) => {
      if (!state || !fresh) return
      if (!anyPerson(players)) setRunning(false)
      const all = [...state.turns.map((t) => t.actions), ...future]
      const to = Math.max(-1, Math.min(index, all.length - 1))
      setLine({ start: line.start, actions: all.slice(0, to + 1).flat() })
      setFuture(all.slice(to + 1))
    },
    [state, fresh, players, line, future],
  )

  /** One turn forward: an undone turn, or else a computer's move. */
  const forward = useCallback(() => {
    if (!anyPerson(players)) setRunning(false)
    if (future.length) redo()
    else setStep(true)
  }, [players, future, redo])

  /** Take back a turn (two against a computer), forgetting it. */
  const takeBack = useCallback(() => {
    if (!state || !fresh) return
    const depth = undoDepth(state.turns, players)
    if (!depth) return
    if (!anyPerson(players)) setRunning(false)
    const index = state.turns.length - depth
    const back = rewind(line.actions, [], state.turns, index)
    setLine({ start: line.start, actions: back.actions })
    setFuture([])
  }, [state, fresh, players, line])

  /** Start again from the standard start, or from a FEN. A set-up
   * position starts paused, so it can be studied before anyone moves. */
  const reset = useCallback(
    (start: string | null = null, paused = false) => {
      setRunning(!paused && anyPerson(players))
      setLine({ start, actions: [] })
      setFuture([])
    },
    [players],
  )

  const turns = state?.turns.length ?? 0
  return {
    line,
    future,
    state,
    fresh,
    canMove,
    error,
    players,
    choosePlayers,
    running,
    setRunning,
    thinking,
    delay,
    setDelay,
    play,
    goto,
    reset,
    takeBack,
    forward,
    /** One turn back. */
    back: () => goto(turns - 2),
    /** To the starting position. */
    toStart: () => goto(-1),
    /** To the last turn, undone ones included. */
    toEnd: () => goto(turns + future.length - 1),
  }
}

export type Game = ReturnType<typeof useGame>
