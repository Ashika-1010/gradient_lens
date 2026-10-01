import { params } from './network.js'

/**
 * Resets the gradients of all trainable parameters in the network to zero.
 *
 * @param {Object} net - Network object containing trainable parameters
 */
export function zeroGrad(net) {
  for (const p of params(net)) {
    p.grad = 0
  }
}

/**
 * Performs a single step of Stochastic Gradient Descent (SGD) on all parameters.
 * Updates parameter values according to the rule: p.data -= learningRate * p.grad.
 * Gradients are preserved and not modified or zeroed.
 *
 * @param {Object} net - Network object containing trainable parameters
 * @param {number} learningRate - Step size scaling factor
 */
export function sgdStep(net, learningRate) {
  for (const p of params(net)) {
    p.data -= learningRate * p.grad
  }
}
