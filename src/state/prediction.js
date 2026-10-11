import { EDGES } from '../visualization/layout.js'

/**
 * Predictable parameter labels: exactly the six trainable edge weights connecting network layers.
 * Biases are explicitly excluded and not predictable.
 */
export const PREDICTABLE_LABELS = Object.freeze(EDGES.map((e) => e.weightLabel))

/**
 * The initial parameter gradient shown as a fully worked example without requiring user prediction.
 */
export const WORKED_EXAMPLE_LABEL = 'w_o_h1'

/**
 * Valid gradient-descent update directions.
 */
export const UPDATE_DIRECTIONS = Object.freeze(['increase', 'decrease', 'same'])

/**
 * Tolerance threshold for floating-point zero gradient comparisons.
 */
const ZERO_TOLERANCE = 1e-9

/**
 * Determines the mathematically correct direction of a weight update under gradient descent:
 * w_new = w - η · g
 *
 * - Positive gradient (g > 1e-9): weight decreases
 * - Negative gradient (g < -1e-9): weight increases
 * - Approximately zero gradient (|g| <= 1e-9): weight stays the same
 *
 * @param {number} grad - Computed parameter gradient
 * @returns {'increase'|'decrease'|'same'|null}
 */
export function correctDirection(grad) {
  if (typeof grad !== 'number' || Number.isNaN(grad)) {
    return null
  }
  if (Math.abs(grad) <= ZERO_TOLERANCE) {
    return 'same'
  }
  if (grad > 0) {
    return 'decrease'
  }
  return 'increase'
}

/**
 * Evaluates whether a user's guess matches the analytical update direction.
 *
 * @param {string} guess - User's guessed direction ('increase'|'decrease'|'same')
 * @param {number} grad - Analytical or autodiff parameter gradient
 * @returns {boolean}
 */
export function isCorrect(guess, grad) {
  const expected = correctDirection(grad)
  return Boolean(guess && expected && guess === expected)
}

/**
 * Checks whether a prediction is required before revealing a parameter's gradient.
 * Worked example (w_o_h1) and biases never require prediction.
 *
 * @param {string} label - Parameter label
 * @returns {boolean}
 */
export function isPredictionRequired(label) {
  return PREDICTABLE_LABELS.includes(label) && label !== WORKED_EXAMPLE_LABEL
}

/**
 * Evaluates whether a parameter's gradient should be visible given both:
 * 1. The backward-pass sequential stage gate (stageRevealed)
 * 2. The prediction gate (prediction submitted if required)
 *
 * Crucial safety rule: A gradient hidden by the stage gate can NEVER be revealed.
 *
 * @param {string} label - Parameter label
 * @param {boolean} stageRevealed - Whether the backward pass has reached this step
 * @param {Record<string, string>} [predictions={}] - User predictions map
 * @returns {boolean}
 */
export function isGradientRevealed(label, stageRevealed, predictions = {}) {
  if (!stageRevealed) {
    return false
  }
  if (!isPredictionRequired(label)) {
    return true
  }
  return Boolean(predictions && predictions[label] != null)
}

/**
 * Immutably records a user's prediction for a parameter.
 * Once a prediction is submitted for a parameter, it is locked and cannot be overwritten.
 * Invalid directions or parameters not requiring prediction are ignored.
 *
 * @param {Record<string, string>} [predictions={}] - Existing predictions map
 * @param {string} label - Parameter label
 * @param {string} direction - Guessed direction ('increase'|'decrease'|'same')
 * @returns {Record<string, string>} New predictions map
 */
export function recordPrediction(predictions = {}, label, direction) {
  if (!UPDATE_DIRECTIONS.includes(direction)) {
    return predictions
  }
  if (!isPredictionRequired(label)) {
    return predictions
  }
  if (predictions && predictions[label] != null) {
    // Guess is locked once submitted
    return predictions
  }
  return {
    ...predictions,
    [label]: direction,
  }
}

/**
 * Applies the prediction gate to an already stage-gated weights object.
 * Gating can ONLY remove gradient information (set to null); it can never reveal
 * a gradient that was already hidden by the backward pass stage gate.
 * Never mutates stageWeights.
 *
 * @param {Record<string, { value: number, grad: number|null }>|null} stageWeights
 * @param {Record<string, string>} [predictions={}]
 * @returns {Record<string, { value: number, grad: number|null }>|null}
 */
export function gatePredictions(stageWeights, predictions = {}) {
  if (!stageWeights) return null

  const result = {}
  for (const label of Object.keys(stageWeights)) {
    const weight = stageWeights[label]
    const stageRevealed = weight.grad !== null
    const revealed = isGradientRevealed(label, stageRevealed, predictions)

    result[label] = {
      value: weight.value,
      grad: revealed ? weight.grad : null,
    }
  }
  return result
}

/**
 * Determines the inspector presentation status for a parameter:
 * - 'locked': stage gate has not reached this parameter yet
 * - 'awaiting-prediction': stage gate has reached this parameter, but prediction has not been submitted
 * - 'revealed': gradient and chain-rule factors are visible (either worked example or prediction submitted)
 *
 * @param {string} label - Parameter label
 * @param {boolean} stageRevealed - Whether the backward pass has reached this step
 * @param {Record<string, string>} [predictions={}] - User predictions map
 * @returns {'locked'|'awaiting-prediction'|'revealed'}
 */
export function inspectorStatus(label, stageRevealed, predictions = {}) {
  if (!label || !stageRevealed) {
    return 'locked'
  }
  if (!isPredictionRequired(label)) {
    return 'revealed'
  }
  if (predictions && predictions[label] != null) {
    return 'revealed'
  }
  return 'awaiting-prediction'
}
