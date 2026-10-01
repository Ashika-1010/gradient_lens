import { forward, params } from './network.js'
import { bceLoss } from './loss.js'
import { zeroGrad } from './optimizer.js'

/**
 * Computes the scalar binary cross-entropy loss for a single example.
 * Evaluates only the forward pass and BCE loss without backward propagation.
 *
 * @param {Object} net - Neural network containing parameters
 * @param {{ input: number[], label: number }} example - Input features and target label
 * @returns {number} Scalar loss value (loss.data)
 */
export function computeLoss(net, example) {
  const out = forward(net, example.input)
  const loss = bceLoss(out.y_hat, example.label)
  return loss.data
}

/**
 * Computes the numerical finite-difference gradient for a specific parameter:
 *   dL/dw ≈ [L(w + eps) - L(w - eps)] / (2 * eps)
 *
 * Does not call backward(), does not inspect or modify param.grad,
 * and guarantees param.data is restored to its exact original value.
 *
 * @param {Object} net - Neural network containing parameters
 * @param {{ input: number[], label: number }} example - Input features and target label
 * @param {Object} param - Parameter Value node whose gradient is being estimated
 * @param {number} [eps=1e-5] - Perturbation step size
 * @returns {number} Numerical gradient estimate
 */
export function numericalGradient(net, example, param, eps = 1e-5) {
  const original = param.data

  param.data = original + eps
  const lossPlus = computeLoss(net, example)

  param.data = original - eps
  const lossMinus = computeLoss(net, example)

  param.data = original

  return (lossPlus - lossMinus) / (2 * eps)
}

/**
 * Computes analytic gradients for all network parameters using reverse-mode autodiff.
 *
 * Sequence: zeroGrad -> forward -> bceLoss -> backward.
 * Does not modify parameter data or perform an optimization step.
 *
 * @param {Object} net - Neural network containing parameters
 * @param {{ input: number[], label: number }} example - Input features and target label
 * @returns {Record<string, number>} Object mapping parameter labels to analytic gradients
 */
export function analyticGradients(net, example) {
  zeroGrad(net)
  const out = forward(net, example.input)
  const loss = bceLoss(out.y_hat, example.label)
  loss.backward()

  const grads = {}
  for (const p of params(net)) {
    grads[p.label] = p.grad
  }
  return grads
}

/**
 * Compares analytic and numerical gradients for all network parameters on a given example.
 *
 * @param {Object} net - Neural network containing parameters
 * @param {{ input: number[], label: number }} example - Input features and target label
 * @param {number} [eps=1e-5] - Perturbation step size for numerical gradient
 * @returns {Array<{ label: string, analytic: number, numerical: number, absDiff: number, absoluteDifference: number }>}
 */
export function compareGradients(net, example, eps = 1e-5) {
  const analytic = analyticGradients(net, example)
  const results = []

  for (const p of params(net)) {
    const num = numericalGradient(net, example, p, eps)
    const ana = analytic[p.label]
    const diff = Math.abs(ana - num)

    results.push({
      label: p.label,
      analytic: ana,
      numerical: num,
      absDiff: diff,
      absoluteDifference: diff,
    })
  }

  return results
}
