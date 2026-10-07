/**
 * Backward pass step definitions and weight gradient gating logic.
 * Controls the sequential reveal of parameter gradients during backpropagation.
 */

export const BACKWARD_STEPS = [
  'w_o_h1',
  'w_o_h2',
  'b_o',
  'w_h1_x1',
  'w_h1_x2',
  'b_h1',
  'w_h2_x1',
  'w_h2_x2',
  'b_h2',
]

export const INITIAL_BACKWARD_STEP = -1
export const MAX_BACKWARD_STEP = BACKWARD_STEPS.length - 1

/**
 * Checks whether stepping forward (next gradient) in the backward pass is permitted.
 *
 * @param {number} backwardStepIndex - Current backward step index
 * @param {boolean} forwardComplete - Whether the forward pass has reached MAX_STEP
 * @returns {boolean}
 */
export function canStepBackward(backwardStepIndex, forwardComplete) {
  return Boolean(forwardComplete && backwardStepIndex < MAX_BACKWARD_STEP)
}

/**
 * Checks whether stepping backward (to previous gradient) in the backward pass is permitted.
 *
 * @param {number} backwardStepIndex - Current backward step index
 * @param {boolean} [forwardComplete=true] - Whether the forward pass has reached MAX_STEP
 * @returns {boolean}
 */
export function canStepPreviousGradient(backwardStepIndex, forwardComplete = true) {
  return Boolean(forwardComplete && backwardStepIndex > INITIAL_BACKWARD_STEP)
}

export const canStepPrevious = canStepPreviousGradient
export const canStepBackwardPass = canStepPreviousGradient

/**
 * Checks whether resetting the backward step sequence is permitted.
 *
 * @param {number} backwardStepIndex - Current backward step index
 * @returns {boolean}
 */
export function canResetBackward(backwardStepIndex) {
  return backwardStepIndex > INITIAL_BACKWARD_STEP
}

/**
 * Returns the parameter key corresponding to the given backward step index.
 *
 * @param {number} backwardStepIndex - Current backward step index
 * @returns {string|null} Parameter key or null if at initial state / out of bounds
 */
export function currentBackwardStepKey(backwardStepIndex) {
  if (
    backwardStepIndex === INITIAL_BACKWARD_STEP ||
    backwardStepIndex < 0 ||
    backwardStepIndex >= BACKWARD_STEPS.length
  ) {
    return null
  }
  return BACKWARD_STEPS[backwardStepIndex]
}

/**
 * Produces a new weights object where every parameter's value is preserved,
 * while gradients are progressively revealed up to backwardStepIndex.
 * Parameter gradients past backwardStepIndex are set to null.
 * Never mutates fullWeights.
 *
 * @param {Object|null} fullWeights - Complete weights object { [label]: { value, grad } }
 * @param {number} backwardStepIndex - Current backward step index (-1 to 8)
 * @returns {Object|null} New gated weights object
 */
export function revealWeightGradients(fullWeights, backwardStepIndex) {
  if (!fullWeights) return null

  const result = {}

  for (const label of Object.keys(fullWeights)) {
    const stepPosition = BACKWARD_STEPS.indexOf(label)
    const revealed = stepPosition !== -1 && stepPosition <= backwardStepIndex

    result[label] = {
      value: fullWeights[label].value,
      grad: revealed ? fullWeights[label].grad : null,
    }
  }

  return result
}
