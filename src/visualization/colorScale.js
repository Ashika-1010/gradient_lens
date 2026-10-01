/**
 * Diverging color scale for visualizing weights, gradients, and activations.
 * Positive values lean blue, negative values lean orange, and zero is neutral gray.
 */

export const POSITIVE = [37, 99, 235]  // Blue
export const NEGATIVE = [234, 88, 12]  // Orange
export const NEUTRAL  = [209, 213, 219] // Neutral Gray

/**
 * Maps a numeric value to an interpolated RGB color string.
 *
 * @param {number} value - Scalar value to map
 * @param {number} [maxMagnitude=1] - Value at which color reaches full saturation
 * @returns {string} CSS rgb(r, g, b) string
 */
export function valueToColor(value, maxMagnitude = 1) {
  if (value === 0 || maxMagnitude <= 0) {
    return `rgb(${NEUTRAL[0]}, ${NEUTRAL[1]}, ${NEUTRAL[2]})`
  }

  const target = value > 0 ? POSITIVE : NEGATIVE
  const rawIntensity = Math.abs(value) / maxMagnitude
  const intensity = Math.min(Math.max(rawIntensity, 0), 1)

  const r = Math.round(NEUTRAL[0] + (target[0] - NEUTRAL[0]) * intensity)
  const g = Math.round(NEUTRAL[1] + (target[1] - NEUTRAL[1]) * intensity)
  const b = Math.round(NEUTRAL[2] + (target[2] - NEUTRAL[2]) * intensity)

  return `rgb(${r}, ${g}, ${b})`
}
