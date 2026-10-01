import { describe, it, expect } from 'vitest'
import { initNetwork, params } from '../../src/engine/network.js'
import { XOR_DATA } from '../../src/engine/xorData.js'
import {
  computeLoss,
  numericalGradient,
  analyticGradients,
  compareGradients,
} from '../../src/engine/gradCheck.js'

describe('Numerical Gradient Checking (Milestone 7)', () => {
  const fixedExample = { input: [1, 0], label: 1 }

  // 1. numericalGradient matches known hand-calculated gradient for w_o_h1
  it('computes numerical gradient matching the known hand-calculated gradient (~ -0.2927) for w_o_h1', () => {
    const net = initNetwork()
    const numGrad = numericalGradient(net, fixedExample, net.o.w1, 1e-5)

    expect(numGrad).toBeCloseTo(-0.2927, 4)
  })

  // 2. numericalGradient restores param.data exactly after calculation
  it('restores param.data exactly to its original value after numerical gradient estimation', () => {
    const net = initNetwork()
    const originalWeight = net.o.w1.data

    numericalGradient(net, fixedExample, net.o.w1, 1e-5)

    expect(net.o.w1.data).toBe(originalWeight)
  })

  // 3. numericalGradient does not alter other parameters
  it('does not alter any other network parameters during calculation', () => {
    const net = initNetwork()
    const initialValues = params(net).map((p) => p.data)

    numericalGradient(net, fixedExample, net.h1.w1, 1e-5)

    const currentValues = params(net).map((p) => p.data)
    expect(currentValues).toEqual(initialValues)
  })

  // 4. analyticGradients returns gradients for all 9 parameters
  it('returns analytic gradients for all 9 named parameters', () => {
    const net = initNetwork()
    const grads = analyticGradients(net, fixedExample)

    const expectedLabels = [
      'w_h1_x1',
      'w_h1_x2',
      'b_h1',
      'w_h2_x1',
      'w_h2_x2',
      'b_h2',
      'w_o_h1',
      'w_o_h2',
      'b_o',
    ]

    expect(Object.keys(grads).sort()).toEqual(expectedLabels.sort())
    for (const label of expectedLabels) {
      expect(typeof grads[label]).toBe('number')
      expect(Number.isFinite(grads[label])).toBe(true)
    }
  })

  // 5. analyticGradients does not modify parameter data
  it('does not modify any parameter data during analytic gradient computation', () => {
    const net = initNetwork()
    const initialValues = params(net).map((p) => p.data)

    analyticGradients(net, fixedExample)

    const currentValues = params(net).map((p) => p.data)
    expect(currentValues).toEqual(initialValues)
  })

  // 6. Compare all 9 parameters on the fixed fixture ([1, 0], label 1)
  it('matches analytic and numerical gradients within 1e-4 for all 9 parameters on fixed fixture', () => {
    const net = initNetwork()
    const comparisons = compareGradients(net, fixedExample, 1e-5)

    expect(comparisons).toHaveLength(9)
    for (const comp of comparisons) {
      expect(comp.absDiff).toBeLessThan(1e-4)
      expect(comp.absoluteDifference).toBeLessThan(1e-4)
    }
  })

  // 7. Compare gradients on all 4 XOR examples
  it('matches analytic and numerical gradients within 1e-4 across all 4 XOR examples', () => {
    for (const sample of XOR_DATA) {
      const net = initNetwork()
      const comparisons = compareGradients(net, sample, 1e-5)

      for (const comp of comparisons) {
        expect(comp.absDiff).toBeLessThan(1e-4)
      }
    }
  })

  // 8. Change every parameter by +0.1, then compare numerical and analytic gradients
  it('maintains gradient match within 1e-4 after perturbing all weights by +0.1', () => {
    const net = initNetwork()

    // Shift all 9 parameters by +0.1
    for (const p of params(net)) {
      p.data += 0.1
    }

    for (const sample of XOR_DATA) {
      const comparisons = compareGradients(net, sample, 1e-5)
      for (const comp of comparisons) {
        expect(comp.absDiff).toBeLessThan(1e-4)
      }
    }
  })

  // 9. Test epsilon behavior (accuracy vs instability)
  it('demonstrates that eps = 1e-5 is accurate, while eps = 0.5 and eps = 1e-14 show reduced accuracy/instability', () => {
    const net = initNetwork()
    const targetParam = net.o.w1
    const analyticVal = analyticGradients(net, fixedExample)[targetParam.label]

    // Optimal epsilon (1e-5)
    const numOpt = numericalGradient(net, fixedExample, targetParam, 1e-5)
    const errOpt = Math.abs(analyticVal - numOpt)

    // Coarse/large epsilon (0.5) introduces O(eps^2) truncation error
    const numLarge = numericalGradient(net, fixedExample, targetParam, 0.5)
    const errLarge = Math.abs(analyticVal - numLarge)

    // Extremely small epsilon (1e-14) suffers from floating-point roundoff / catastrophic cancellation
    const numTiny = numericalGradient(net, fixedExample, targetParam, 1e-14)
    const errTiny = Math.abs(analyticVal - numTiny)

    // Optimal epsilon is highly accurate
    expect(errOpt).toBeLessThan(1e-4)

    // Coarse epsilon error is noticeably larger than optimal epsilon error
    expect(errLarge).toBeGreaterThan(errOpt)

    // Sub-epsilon / tiny epsilon experiences roundoff error significantly worse than 1e-5
    expect(errTiny).toBeGreaterThan(errOpt)
  })
})
