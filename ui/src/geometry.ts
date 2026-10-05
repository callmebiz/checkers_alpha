// Board geometry: PDN squares (1-32) to rows, columns and drawing
// coordinates, matching core/game.py. Row 0 is Black's back row (squares
// 1-4); dark squares are those where row + column is odd.

/** Size of one board square in SVG units. */
export const CELL = 100

/** Row and column (0-7) of a PDN square. */
export function squareToRowCol(square: number): [number, number] {
  const s = square - 1
  const row = s >> 2
  return [row, 2 * (s & 3) + (1 - (row & 1))]
}

/** The PDN square at a row and column, or null for a light square. */
export function rowColToSquare(row: number, col: number): number | null {
  if (row < 0 || row > 7 || col < 0 || col > 7) return null
  if ((row + col) % 2 === 0) return null
  return row * 4 + Math.floor(col / 2) + 1
}

/** Centre of a square in SVG units. With `flipped`, Black's side is at the
 * bottom (the board is rotated 180 degrees). */
export function squareCenter(
  square: number,
  flipped: boolean,
): [number, number] {
  let [row, col] = squareToRowCol(square)
  if (flipped) [row, col] = [7 - row, 7 - col]
  return [(col + 0.5) * CELL, (row + 0.5) * CELL]
}

/** Points of an arrow's shaft, from one square's centre toward another,
 * shortened at both ends so it doesn't hide the pieces. */
export function arrowEnds(
  from: number,
  to: number,
  flipped: boolean,
  trim = 0.28 * CELL,
): [number, number, number, number] {
  const [x1, y1] = squareCenter(from, flipped)
  const [x2, y2] = squareCenter(to, flipped)
  const length = Math.hypot(x2 - x1, y2 - y1)
  const ux = (x2 - x1) / length
  const uy = (y2 - y1) / length
  return [
    x1 + ux * trim * 0.6,
    y1 + uy * trim * 0.6,
    x2 - ux * trim,
    y2 - uy * trim,
  ]
}
