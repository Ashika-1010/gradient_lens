import { Value } from './value.js'

export const EPS = 1e-7

/**
 * Computes Binary Cross-Entropy (BCE) loss between predictions yHat and targets y:
 * L = - [ y * log(p) + (1 - y) * log(1 - p) ]
 * where p = yHat.clamp(EPS, 1 - EPS)
 *
 * Implemented purely using Value autodiff operations (clamp, log, neg, add, mul, sub)
 * so that calling loss.backward() propagates real gradients through y_hat and
 * the underlying neural network parameters.
 *
 * @param {Value|number} yHat - Predicted probability (output of sigmoid)
 * @param {Value|number} y - Target ground-truth label (0 or 1)
 * @returns {Value} The loss node in the autodiff computation graph labeled 'L'
 */
export function bceLoss(yHat, y) {
  const yHatVal = Value.from(yHat, 'y_hat')
  const p = yHatVal.clamp(EPS, 1 - EPS)

  const term1 = Value.from(y).mul(p.log())

  const term2 = Value.from(1 - y).mul(
    Value.from(1).sub(p).log()
  )

  const loss = term1.add(term2).neg()
  loss.label = 'L'

  return loss
}

export const binaryCrossEntropy = bceLoss
export const bce = bceLoss
