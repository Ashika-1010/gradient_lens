/**
 * Analytical chain-rule expansion and decomposition for Gradient Lens.
 * Deconstructs the backpropagation gradient of each parameter into individual symbolic
 * factors and numerical products, cross-checking against the autodiff engine.
 */

/**
 * Metadata definitions for all 9 trainable network parameters.
 */
export const PARAM_META = {
  w_o_h1: {
    label: 'w_o_h1',
    name: 'w_o_h1',
    displayName: 'Output Weight w_o_h1',
    layer: 'output',
    type: 'weight',
    from: 'h1',
    to: 'o',
    outputWeight: null,
    formula: '(ŷ - y) * a_h1',
    description: 'Output weight connecting hidden neuron h1 to output neuron o',
  },
  w_o_h2: {
    label: 'w_o_h2',
    name: 'w_o_h2',
    displayName: 'Output Weight w_o_h2',
    layer: 'output',
    type: 'weight',
    from: 'h2',
    to: 'o',
    outputWeight: null,
    formula: '(ŷ - y) * a_h2',
    description: 'Output weight connecting hidden neuron h2 to output neuron o',
  },
  b_o: {
    label: 'b_o',
    name: 'b_o',
    displayName: 'Output Bias b_o',
    layer: 'output',
    type: 'bias',
    from: null,
    to: 'o',
    outputWeight: null,
    formula: '(ŷ - y)',
    description: 'Bias for output neuron o',
  },
  w_h1_x1: {
    label: 'w_h1_x1',
    name: 'w_h1_x1',
    displayName: 'Hidden Weight w_h1_x1',
    layer: 'hidden',
    type: 'weight',
    from: 'x1',
    to: 'h1',
    hiddenNeuron: 'h1',
    outputWeight: 'w_o_h1',
    formula: "(ŷ - y) * w_o_h1 * σ'(z_h1) * x1",
    description: 'Hidden weight connecting input x1 to hidden neuron h1',
  },
  w_h1_x2: {
    label: 'w_h1_x2',
    name: 'w_h1_x2',
    displayName: 'Hidden Weight w_h1_x2',
    layer: 'hidden',
    type: 'weight',
    from: 'x2',
    to: 'h1',
    hiddenNeuron: 'h1',
    outputWeight: 'w_o_h1',
    formula: "(ŷ - y) * w_o_h1 * σ'(z_h1) * x2",
    description: 'Hidden weight connecting input x2 to hidden neuron h1',
  },
  b_h1: {
    label: 'b_h1',
    name: 'b_h1',
    displayName: 'Hidden Bias b_h1',
    layer: 'hidden',
    type: 'bias',
    from: null,
    to: 'h1',
    hiddenNeuron: 'h1',
    outputWeight: 'w_o_h1',
    formula: "(ŷ - y) * w_o_h1 * σ'(z_h1)",
    description: 'Bias for hidden neuron h1',
  },
  w_h2_x1: {
    label: 'w_h2_x1',
    name: 'w_h2_x1',
    displayName: 'Hidden Weight w_h2_x1',
    layer: 'hidden',
    type: 'weight',
    from: 'x1',
    to: 'h2',
    hiddenNeuron: 'h2',
    outputWeight: 'w_o_h2',
    formula: "(ŷ - y) * w_o_h2 * σ'(z_h2) * x1",
    description: 'Hidden weight connecting input x1 to hidden neuron h2',
  },
  w_h2_x2: {
    label: 'w_h2_x2',
    name: 'w_h2_x2',
    displayName: 'Hidden Weight w_h2_x2',
    layer: 'hidden',
    type: 'weight',
    from: 'x2',
    to: 'h2',
    hiddenNeuron: 'h2',
    outputWeight: 'w_o_h2',
    formula: "(ŷ - y) * w_o_h2 * σ'(z_h2) * x2",
    description: 'Hidden weight connecting input x2 to hidden neuron h2',
  },
  b_h2: {
    label: 'b_h2',
    name: 'b_h2',
    displayName: 'Hidden Bias b_h2',
    layer: 'hidden',
    type: 'bias',
    from: null,
    to: 'h2',
    hiddenNeuron: 'h2',
    outputWeight: 'w_o_h2',
    formula: "(ŷ - y) * w_o_h2 * σ'(z_h2)",
    description: 'Bias for hidden neuron h2',
  },
}

/**
 * Retrieves the parameter Value node from the network given its label.
 *
 * @param {Object} net - Network object returned by initNetwork
 * @param {string} label - Parameter label
 * @returns {import('./value.js').Value|null}
 */
export function getParamNode(net, label) {
  if (!net) return null
  switch (label) {
    case 'w_h1_x1': return net.h1.w1
    case 'w_h1_x2': return net.h1.w2
    case 'b_h1':    return net.h1.b
    case 'w_h2_x1': return net.h2.w1
    case 'w_h2_x2': return net.h2.w2
    case 'b_h2':    return net.h2.b
    case 'w_o_h1':  return net.o.w1
    case 'w_o_h2':  return net.o.w2
    case 'b_o':     return net.o.b
    default:        return null
  }
}

/**
 * Constructs an independent analytical chain-rule expression breakdown for a parameter.
 * Returns serializable data containing label, formula, factors, product, and engineGrad.
 * Never mutates the autodiff engine.
 *
 * @param {Object} net - Network object containing Value parameter instances
 * @param {Object} out - Forward pass output object from forward()
 * @param {string} label - Parameter label
 * @param {number} y - Ground truth target label (0 or 1)
 * @returns {Object|null} Chain expression breakdown or null if invalid
 */
export function buildChainExpression(net, out, label, y) {
  const meta = PARAM_META[label]
  if (!meta || !net || !out) return null

  const paramNode = getParamNode(net, label)
  if (!paramNode) return null

  // ∂L/∂z_o = ŷ - y (simplified binary cross-entropy + sigmoid derivative)
  const dL_dzo = out.y_hat.data - y
  const errorFactor = {
    label: '(ŷ - y)',
    name: '∂L/∂z_o',
    symbol: '(ŷ - y)',
    description: 'Output prediction error',
    value: dL_dzo,
  }

  let factors = []

  if (label === 'w_o_h1') {
    factors = [
      errorFactor,
      {
        label: 'a_h1',
        name: '∂z_o/∂w_o_h1',
        symbol: 'a_h1',
        description: 'Hidden activation h1',
        value: out.a_h1.data,
      },
    ]
  } else if (label === 'w_o_h2') {
    factors = [
      errorFactor,
      {
        label: 'a_h2',
        name: '∂z_o/∂w_o_h2',
        symbol: 'a_h2',
        description: 'Hidden activation h2',
        value: out.a_h2.data,
      },
    ]
  } else if (label === 'b_o') {
    factors = [
      errorFactor,
    ]
  } else if (label === 'w_h1_x1') {
    const sigPrime = out.a_h1.data * (1 - out.a_h1.data)
    factors = [
      errorFactor,
      {
        label: 'w_o_h1',
        name: '∂z_o/∂a_h1',
        symbol: 'w_o_h1',
        description: 'Downstream output weight w_o_h1',
        value: net.o.w1.data,
      },
      {
        label: "σ'(z_h1)",
        name: '∂a_h1/∂z_h1',
        symbol: "σ'(z_h1)",
        description: 'Sigmoid derivative at h1',
        value: sigPrime,
      },
      {
        label: 'x1',
        name: '∂z_h1/∂w_h1_x1',
        symbol: 'x1',
        description: 'Input feature x1',
        value: out.x1.data,
      },
    ]
  } else if (label === 'w_h1_x2') {
    const sigPrime = out.a_h1.data * (1 - out.a_h1.data)
    factors = [
      errorFactor,
      {
        label: 'w_o_h1',
        name: '∂z_o/∂a_h1',
        symbol: 'w_o_h1',
        description: 'Downstream output weight w_o_h1',
        value: net.o.w1.data,
      },
      {
        label: "σ'(z_h1)",
        name: '∂a_h1/∂z_h1',
        symbol: "σ'(z_h1)",
        description: 'Sigmoid derivative at h1',
        value: sigPrime,
      },
      {
        label: 'x2',
        name: '∂z_h1/∂w_h1_x2',
        symbol: 'x2',
        description: 'Input feature x2',
        value: out.x2.data,
      },
    ]
  } else if (label === 'b_h1') {
    const sigPrime = out.a_h1.data * (1 - out.a_h1.data)
    factors = [
      errorFactor,
      {
        label: 'w_o_h1',
        name: '∂z_o/∂a_h1',
        symbol: 'w_o_h1',
        description: 'Downstream output weight w_o_h1',
        value: net.o.w1.data,
      },
      {
        label: "σ'(z_h1)",
        name: '∂a_h1/∂z_h1',
        symbol: "σ'(z_h1)",
        description: 'Sigmoid derivative at h1',
        value: sigPrime,
      },
    ]
  } else if (label === 'w_h2_x1') {
    const sigPrime = out.a_h2.data * (1 - out.a_h2.data)
    factors = [
      errorFactor,
      {
        label: 'w_o_h2',
        name: '∂z_o/∂a_h2',
        symbol: 'w_o_h2',
        description: 'Downstream output weight w_o_h2',
        value: net.o.w2.data,
      },
      {
        label: "σ'(z_h2)",
        name: '∂a_h2/∂z_h2',
        symbol: "σ'(z_h2)",
        description: 'Sigmoid derivative at h2',
        value: sigPrime,
      },
      {
        label: 'x1',
        name: '∂z_h2/∂w_h2_x1',
        symbol: 'x1',
        description: 'Input feature x1',
        value: out.x1.data,
      },
    ]
  } else if (label === 'w_h2_x2') {
    const sigPrime = out.a_h2.data * (1 - out.a_h2.data)
    factors = [
      errorFactor,
      {
        label: 'w_o_h2',
        name: '∂z_o/∂a_h2',
        symbol: 'w_o_h2',
        description: 'Downstream output weight w_o_h2',
        value: net.o.w2.data,
      },
      {
        label: "σ'(z_h2)",
        name: '∂a_h2/∂z_h2',
        symbol: "σ'(z_h2)",
        description: 'Sigmoid derivative at h2',
        value: sigPrime,
      },
      {
        label: 'x2',
        name: '∂z_h2/∂w_h2_x2',
        symbol: 'x2',
        description: 'Input feature x2',
        value: out.x2.data,
      },
    ]
  } else if (label === 'b_h2') {
    const sigPrime = out.a_h2.data * (1 - out.a_h2.data)
    factors = [
      errorFactor,
      {
        label: 'w_o_h2',
        name: '∂z_o/∂a_h2',
        symbol: 'w_o_h2',
        description: 'Downstream output weight w_o_h2',
        value: net.o.w2.data,
      },
      {
        label: "σ'(z_h2)",
        name: '∂a_h2/∂z_h2',
        symbol: "σ'(z_h2)",
        description: 'Sigmoid derivative at h2',
        value: sigPrime,
      },
    ]
  }

  const rawProduct = factors.reduce((acc, f) => acc * f.value, 1)
  const product = rawProduct === 0 ? 0 : rawProduct
  const engineGrad = paramNode.grad === 0 ? 0 : paramNode.grad

  return {
    label,
    paramValue: paramNode.data,
    formula: meta.formula,
    factors,
    product,
    engineGrad,
    diff: engineGrad !== null && engineGrad !== undefined ? Math.abs(product - engineGrad) : null,
    agrees: engineGrad !== null && engineGrad !== undefined ? Math.abs(product - engineGrad) < 1e-6 : false,
  }
}
