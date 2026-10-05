// Playback controls under the board, as in chess apps: to the start, one
// turn back, play/pause, one turn forward (or one computer move), to the
// end. Plus the speed of play, taking a move back and a new game.
import { useState } from 'react'
import type { Color } from '../api'
import type { Game } from '../game'

// the standard start with White to move (Black moves first by the rules)
const WHITE_FIRST = 'W:W21-32:B1-12'

const SPEEDS = [
  { ms: 1500, label: 'slow' },
  { ms: 500, label: 'normal' },
  { ms: 150, label: 'fast' },
]

/** Icons drawn on a 24-unit grid. */
const ICONS = {
  start: 'M6 5h2.5v14H6zM19 5v14l-9-7z',
  back: 'M15.5 5 8.5 12l7 7',
  play: 'M8 5v14l11-7z',
  pause: 'M7 5h3.5v14H7zM13.5 5H17v14h-3.5z',
  forward: 'M8.5 5l7 7-7 7',
  end: 'M15.5 5H18v14h-2.5zM5 5v14l9-7z',
}

/** One icon button. */
function IconButton(props: {
  icon: keyof typeof ICONS
  label: string
  disabled?: boolean
  onClick: () => void
}) {
  const stroked = props.icon === 'back' || props.icon === 'forward'
  return (
    <button
      className="nav"
      onClick={props.onClick}
      disabled={props.disabled}
      title={props.label}
      aria-label={props.label}
    >
      <svg viewBox="0 0 24 24" className={stroked ? 'stroked' : 'filled'}>
        <path d={ICONS[props.icon]} />
      </svg>
    </button>
  )
}

/** The control bar. */
export function Controls(props: {
  game: Game
  onFlip: () => void
  onEdit: () => void
}) {
  const { game } = props
  const [first, setFirst] = useState<Color>('black')
  const { state, future, running, players } = game
  const atStart = !state?.turns.length
  const computerToMove =
    !!state && !state.result && players[state.turn].kind !== 'human'
  // forward has something to do: an undone turn, or a computer to move
  const canStep = future.length > 0 || computerToMove
  return (
    <div className="controls">
      <div className="nav-row">
        <IconButton
          icon="start"
          label="Start (Home)"
          disabled={atStart}
          onClick={game.toStart}
        />
        <IconButton
          icon="back"
          label="Back (←)"
          disabled={atStart}
          onClick={game.back}
        />
        <IconButton
          icon={running ? 'pause' : 'play'}
          label={
            running
              ? 'Pause: computers wait, move either side (space)'
              : 'Play: computers move (space)'
          }
          onClick={() => game.setRunning(!running)}
        />
        <IconButton
          icon="forward"
          label="Forward, or one computer move (→)"
          disabled={!canStep}
          onClick={game.forward}
        />
        <IconButton
          icon="end"
          label="End (End)"
          disabled={!future.length}
          onClick={game.toEnd}
        />
      </div>
      <div className="buttons">
        <select
          value={game.delay}
          onChange={(e) => game.setDelay(+e.target.value)}
          title="Speed of play"
        >
          {SPEEDS.map((s) => (
            <option key={s.ms} value={s.ms}>
              {s.label}
            </option>
          ))}
        </select>
        <button onClick={props.onEdit}>Edit position</button>
        <button onClick={props.onFlip} title="Flip the board (F)">
          Flip
        </button>
        <button onClick={game.takeBack} disabled={atStart}>
          Take back
        </button>
        <select
          value={first}
          onChange={(e) => setFirst(e.target.value as Color)}
          title="Who moves first in a new game"
        >
          <option value="black">Black starts</option>
          <option value="white">White starts</option>
        </select>
        <button
          onClick={() => game.reset(first === 'white' ? WHITE_FIRST : null)}
        >
          New game
        </button>
      </div>
    </div>
  )
}
