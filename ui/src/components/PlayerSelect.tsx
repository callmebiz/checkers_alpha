// Choosing who plays a side (or which model analyses): a person, any
// checkpoint with a search budget, or a plain engine.
import type { Players as Catalog } from '../api'
import type { Player } from '../game'

/** A model checkpoint picker, grouped by run (newest run first). */
export function ModelSelect(props: {
  catalog: Catalog
  value: string
  onChange: (model: string) => void
}) {
  return (
    <select
      value={props.value}
      onChange={(e) => props.onChange(e.target.value)}
    >
      {props.catalog.runs.map((run) => (
        <optgroup key={run.run} label={`${run.run} (${run.shape})`}>
          {[...run.checkpoints].reverse().map((c) => (
            <option key={c.model} value={c.model}>
              {checkpointLabel(run.run, run.shape, c.iteration, c.hours)}
            </option>
          ))}
        </optgroup>
      ))}
    </select>
  )
}

/** How a checkpoint reads in a list: run, shape, iteration, hours. */
export function checkpointLabel(
  run: string,
  shape: string,
  iteration: number,
  hours: number | null,
): string {
  const name = run.split('/').at(-1)
  const label = name === shape ? shape : `${name} ${shape}`
  const time = hours == null ? '' : `, ${hours.toFixed(2)} h`
  return `${label} · iter ${iteration + 1}${time}`
}

const SIMS = [1, 100, 200, 400, 800, 1600, 3200, 6400]
const ENGINE_NAMES: Record<string, string> = { random: 'random moves' }

/** The player picker for one side. */
export function PlayerSelect(props: {
  label: string
  catalog: Catalog
  value: Player
  onChange: (player: Player) => void
}) {
  const { catalog, value, onChange } = props
  const model = catalog.default_model
  /** Switch kind, keeping sensible defaults for the new kind. */
  function setKind(kind: Player['kind']) {
    if (kind === 'human') onChange({ kind })
    else if (kind === 'engine') onChange({ kind, spec: 'ab:6' })
    else if (model) onChange({ kind, model, sims: 800 })
  }
  return (
    <fieldset className="player">
      <legend>{props.label}</legend>
      <select
        value={value.kind}
        onChange={(e) => setKind(e.target.value as Player['kind'])}
      >
        <option value="human">person</option>
        <option value="model" disabled={!model}>
          network
        </option>
        <option value="engine">engine</option>
      </select>
      {value.kind === 'model' && (
        <>
          <ModelSelect
            catalog={catalog}
            value={value.model}
            onChange={(m) => onChange({ ...value, model: m })}
          />
          <select
            value={value.sims}
            title="Search simulations per hop (1: the raw network)"
            onChange={(e) => onChange({ ...value, sims: +e.target.value })}
          >
            {SIMS.map((n) => (
              <option key={n} value={n}>
                {n === 1 ? 'no search' : `${n} sims`}
              </option>
            ))}
          </select>
        </>
      )}
      {value.kind === 'engine' && (
        <select
          value={value.spec}
          onChange={(e) => onChange({ kind: 'engine', spec: e.target.value })}
        >
          {catalog.engines.map((spec) => (
            <option key={spec} value={spec}>
              {ENGINE_NAMES[spec] ?? `alpha-beta depth ${spec.slice(3)}`}
            </option>
          ))}
        </select>
      )}
    </fieldset>
  )
}
