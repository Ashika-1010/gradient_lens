import { Value } from './value.js'

/**
 * Computes Binary Cross-Entropy (BCE) loss between predictions and target labels:
 * L = - [ y * ln(y_hat) + (1 - y) * ln(1 - y_hat) ]
 *
 * Implemented purely using Value autodiff operations (log, neg, add, mul, sub)
 * so that calling loss.backward() propagates real gradients through y_hat and
 * the underlying neural network parameters.
 *
 * @param {Value|number} yHat - Predicted probability (typically sigmoid output)
 * @param {Value|number} y - Target ground-truth label (0 or 1)
 * @param {number} [eps=1e-15] - Numerical stability epsilon for probability clamping
 * @returns {Value} The loss node in the autodiff computation graph
 */
export function binaryCrossEntropy(yHat, y, eps = 1e-15) {
  const yVal = Value.from(y, 'y')
  const yHatVal = Value.from(yHat, 'y_hat')

  // 1 - y
  const oneMinusY = Value.from(1).sub(yVal)

  // ln(y_hat + eps) to prevent log(0) while keeping yHat connected to the graph
  const logYHat = (eps > 0 ? yHatVal.add(eps) : yHatVal).log()

  // 1 - y_hat
  const oneMinusYHat = Value.from(1).sub(yHatVal)

  // ln(1 - y_hat + eps)
  const logOneMinusYHat = (eps > 0 ? oneMinusYHat.add(eps) : oneMinusYHat).log()

  // y * ln(y_hat)
  const term1 = yVal.mul(logYHat)

  // (1 - y) * ln(1 - y_hat)
  const term2 = oneMinusY.mul(logOneMinusYHat)

  // - [ y * ln(y_hat) + (1 - y) * ln(1 - y_hat) ]
  const loss = term1.add(term2).neg()
  loss.label = 'loss'

  return loss
}

export const bce = binaryCrossEntropy
export const bceLoss = binaryCrossEntropy
