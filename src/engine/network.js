import { Value } from './value.js'

const DEFAULT_WEIGHTS = {
  h1: { w1: 0.5, w2: -0.4, b: 0.1 },
  h2: { w1: 0.3, w2: 0.8, b: -0.2 },
  o:  { w1: 0.7, w2: -0.6, b: 0.05 },
}

function leaf(v, label) {
  return new Value(v, [], '', label)
}

/**
 * Initializes the 2-2-1 fixed architecture neural network.
 * Creates and returns the 9 persistent trainable parameters.
 *
 * @param {Object} [weights=DEFAULT_WEIGHTS] - Initial weight configuration
 * @returns {Object} Network object containing parameter Value nodes
 */
export function initNetwork(weights = DEFAULT_WEIGHTS) {
  return {
    h1: {
      w1: leaf(weights.h1.w1, 'w_h1_x1'),
      w2: leaf(weights.h1.w2, 'w_h1_x2'),
      b:  leaf(weights.h1.b, 'b_h1'),
    },

    h2: {
      w1: leaf(weights.h2.w1, 'w_h2_x1'),
      w2: leaf(weights.h2.w2, 'w_h2_x2'),
      b:  leaf(weights.h2.b, 'b_h2'),
    },

    o: {
      w1: leaf(weights.o.w1, 'w_o_h1'),
      w2: leaf(weights.o.w2, 'w_o_h2'),
      b:  leaf(weights.o.b, 'b_o'),
    },
  }
}

/**
 * Returns an array of all 9 trainable parameter Value instances.
 *
 * @param {Object} net - Network object returned by initNetwork
 * @returns {Value[]} Array of 9 Value objects
 */
export function params(net) {
  return [
    net.h1.w1,
    net.h1.w2,
    net.h1.b,

    net.h2.w1,
    net.h2.w2,
    net.h2.b,

    net.o.w1,
    net.o.w2,
    net.o.b,
  ]
}

/**
 * Performs a forward pass through the 2-2-1 neural network.
 * Creates fresh input leaf nodes and a fresh computation graph while reusing parameter objects.
 *
 * @param {Object} net - Network object containing parameters
 * @param {number[]} input - Array of 2 numerical input features [x1, x2]
 * @returns {Object} Object containing intermediate and final computation graph nodes
 */
export function forward(net, input) {
  const x1 = leaf(input[0], 'x1')
  const x2 = leaf(input[1], 'x2')

  const z_h1 =
    net.h1.w1.mul(x1)
      .add(net.h1.w2.mul(x2))
      .add(net.h1.b)

  z_h1.label = 'z_h1'

  const a_h1 = z_h1.sigmoid()
  a_h1.label = 'a_h1'

  const z_h2 =
    net.h2.w1.mul(x1)
      .add(net.h2.w2.mul(x2))
      .add(net.h2.b)

  z_h2.label = 'z_h2'

  const a_h2 = z_h2.sigmoid()
  a_h2.label = 'a_h2'

  const z_o =
    net.o.w1.mul(a_h1)
      .add(net.o.w2.mul(a_h2))
      .add(net.o.b)

  z_o.label = 'z_o'

  const y_hat = z_o.sigmoid()
  y_hat.label = 'y_hat'

  return {
    x1,
    x2,
    z_h1,
    a_h1,
    z_h2,
    a_h2,
    z_o,
    y_hat,
  }
}
