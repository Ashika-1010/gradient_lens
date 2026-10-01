import { params } from '../engine/network.js'

/**
 * Builds an immutable, plain-primitive snapshot of the network state for UI consumption.
 * Decouples the visualizer from mutable engine Value objects.
 *
 * @param {Object} net - Neural network containing parameters
 * @param {Object|null} [forwardOutput=null] - Output dictionary from forward() or null
 * @returns {{ weights: Record<string, { value: number, grad: number }>, activations: Object|null }}
 */
export function buildSnapshot(net, forwardOutput = null) {
  const weights = {}

  for (const p of params(net)) {
    weights[p.label] = {
      value: p.data,
      grad: p.grad,
    }
  }

  let activations = null

  if (forwardOutput !== null) {
    activations = {
      x1: forwardOutput.x1.data,
      x2: forwardOutput.x2.data,
      z_h1: forwardOutput.z_h1.data,
      a_h1: forwardOutput.a_h1.data,
      z_h2: forwardOutput.z_h2.data,
      a_h2: forwardOutput.a_h2.data,
      z_o: forwardOutput.z_o.data,
      y_hat: forwardOutput.y_hat.data,
    }
  }

  return {
    weights,
    activations,
  }
}
