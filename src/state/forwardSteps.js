/**
 * Forward pass step definitions and activation gating logic.
 * Controls the sequential reveal of forward computation nodes.
 */

export const FORWARD_STEPS = [
  'z_h1',
  'a_h1',
  'z_h2',
  'a_h2',
  'z_o',
  'y_hat',
]

export const INITIAL_STEP = -1
export const MAX_STEP = FORWARD_STEPS.length - 1

/**
 * Checks whether stepping forward is permitted.
 *
 * @param {number} stepIndex - Current step index
 * @returns {boolean}
 */
export function canStepForward(stepIndex) {
  return stepIndex < MAX_STEP
}

/**
 * Checks whether stepping backward (to previous step) is permitted.
 *
 * @param {number} stepIndex - Current step index
 * @returns {boolean}
 */
export function canStepBackward(stepIndex) {
  return stepIndex > INITIAL_STEP
}

export const canStepPrevious = canStepBackward

/**
 * Checks whether resetting the step sequence is permitted.
 *
 * @param {number} stepIndex - Current step index
 * @returns {boolean}
 */
export function canReset(stepIndex) {
  return stepIndex > INITIAL_STEP
}

/**
 * Returns the activation key corresponding to the given step index.
 *
 * @param {number} stepIndex - Current step index
 * @returns {string|null} Activation key or null if at initial state
 */
export function currentStepKey(stepIndex) {
  if (stepIndex === INITIAL_STEP || stepIndex < 0 || stepIndex >= FORWARD_STEPS.length) {
    return null
  }
  return FORWARD_STEPS[stepIndex]
}

/**
 * Produces a new activations object where only steps up to stepIndex are revealed,
 * while unrevealed steps are set to null. Input features x1 and x2 are always preserved.
 * Never mutates fullActivations.
 *
 * @param {Object|null} fullActivations - Complete activations object from forward pass
 * @param {number} stepIndex - Current step index (-1 to 5)
 * @returns {Object|null} New gated activations object
 */
export function revealActivations(fullActivations, stepIndex) {
  if (!fullActivations) return null

  const result = {
    x1: fullActivations.x1,
    x2: fullActivations.x2,
  }

  FORWARD_STEPS.forEach((key, i) => {
    result[key] = i <= stepIndex ? fullActivations[key] : null
  })

  return result
}
