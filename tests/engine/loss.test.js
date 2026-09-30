import { describe, it, expect } from 'vitest'
import { Value, topologicalOrder } from '../../src/engine/value.js'
import { initNetwork, forward, params } from '../../src/engine/network.js'
import { binaryCrossEntropy, bce, bceLoss } from '../../src/engine/loss.js'
import { XOR_DATA } from '../../src/engine/xorData.js'

describe('Binary Cross-Entropy Loss (Milestone 4)', () => {
  // 1. BCE when y=1
  it('computes BCE loss when y = 1 and y_hat = 0.8 as -ln(0.8)', () => {
    const yHat = new Value(0.8, [], '', 'y_hat')
    const loss = binaryCrossEntropy(yHat, 1)

    // Expected loss = -ln(0.8) ≈ 0.22314355
    const expected = -Math.log(0.8)
    expect(loss.data).toBeCloseTo(expected, 10)
    expect(loss.data).toBeCloseTo(0.22314355, 6)
  })

  // 2. BCE when y=0
  it('computes BCE loss when y = 0 and y_hat = 0.2 as -ln(0.8)', () => {
    const yHat = new Value(0.2, [], '', 'y_hat')
    const loss = binaryCrossEntropy(yHat, 0)

    // Expected loss = -ln(1 - 0.2) = -ln(0.8) ≈ 0.22314355
    const expected = -Math.log(0.8)
    expect(loss.data).toBeCloseTo(expected, 10)
    expect(loss.data).toBeCloseTo(0.22314355, 6)
  })

  // 3. BCE gradient with respect to y_hat
  it('computes exact analytical gradient dL/dy_hat for both y = 1 and y = 0', () => {
    // When y = 1: dL/dy_hat = -1/y_hat
    const yHat1 = new Value(0.8, [], '', 'y_hat1')
    const loss1 = binaryCrossEntropy(yHat1, 1)
    loss1.backward()
    // Analytical gradient: -1 / 0.8 = -1.25
    expect(yHat1.grad).toBeCloseTo(-1.25, 10)

    // When y = 0: dL/dy_hat = 1 / (1 - y_hat)
    const yHat0 = new Value(0.2, [], '', 'y_hat0')
    const loss0 = binaryCrossEntropy(yHat0, 0)
    loss0.backward()
    // Analytical gradient: 1 / (1 - 0.2) = 1 / 0.8 = 1.25
    expect(yHat0.grad).toBeCloseTo(1.25, 10)

    // General case: dL/dy_hat = (y_hat - y) / (y_hat * (1 - y_hat))
    const yHatGen = new Value(0.65, [], '', 'y_hat_gen')
    const lossGen = binaryCrossEntropy(yHatGen, 1)
    lossGen.backward()
    const expectedGrad = (0.65 - 1.0) / (0.65 * (1.0 - 0.65))
    expect(yHatGen.grad).toBeCloseTo(expectedGrad, 10)
  })

  // 4. Full network loss for XOR sample (1, 0) with y = 1
  it('propagates loss through full 2-2-1 network for input [1, 0] with y = 1', () => {
    const net = initNetwork()
    const out = forward(net, [1, 0])
    const loss = binaryCrossEntropy(out.y_hat, 1)

    // Verify forward loss value ≈ 0.6040306
    expect(loss.data).toBeCloseTo(0.6040306, 4)

    loss.backward()

    // Verify output pre-activation gradient: dL/dz_o = y_hat - y ≈ 0.546604 - 1 = -0.453396
    expect(out.z_o.grad).toBeCloseTo(-0.453396, 5)
    expect(out.z_o.grad).toBeCloseTo(out.y_hat.data - 1.0, 8)

    // Verify all 9 parameters receive finite gradients
    const allParams = params(net)
    expect(allParams).toHaveLength(9)

    for (const p of allParams) {
      expect(Number.isFinite(p.grad)).toBe(true)
      expect(Number.isNaN(p.grad)).toBe(false)
    }
  })

  // 5. Hand-check output weight gradient w_o_h1 ≈ -0.29274
  it('computes correct hand-calculated gradient for output weight w_o_h1 of approximately -0.29274', () => {
    const net = initNetwork()
    const out = forward(net, [1, 0])
    const loss = binaryCrossEntropy(out.y_hat, 1)

    loss.backward()

    // dL/dw_o_h1 = (dL/dz_o) * a_h1 ≈ -0.4533958 * 0.6456563 ≈ -0.29274
    expect(net.o.w1.grad).toBeCloseTo(-0.29274, 4)

    // Verify other key intermediate parameter gradients
    expect(net.o.w2.grad).toBeCloseTo(-0.23802, 4)
    expect(net.o.b.grad).toBeCloseTo(-0.45340, 4)
  })

  // 6. Verify gradients are never NaN or Infinity across all XOR dataset samples
  it('ensures gradients are finite and never NaN or Infinity across all 4 XOR samples', () => {
    for (const sample of XOR_DATA) {
      const net = initNetwork()
      const out = forward(net, sample.input)
      const loss = binaryCrossEntropy(out.y_hat, sample.label)

      loss.backward()

      expect(Number.isFinite(loss.data)).toBe(true)
      expect(Number.isNaN(loss.data)).toBe(false)

      const allParams = params(net)
      for (const p of allParams) {
        expect(Number.isFinite(p.grad)).toBe(true)
        expect(Number.isNaN(p.grad)).toBe(false)
      }
    }
  })

  // 7. Graph participation and structure
  it('creates a Value node that participates properly in the computation graph', () => {
    const net = initNetwork()
    const out = forward(net, [1, 0])
    const loss = binaryCrossEntropy(out.y_hat, 1)

    expect(loss).toBeInstanceOf(Value)
    expect(loss.label).toBe('loss')
    expect(loss.op).toBe('neg')

    // Verify graph connectivity from loss back to parameters and inputs
    const topo = topologicalOrder(loss)
    expect(topo[topo.length - 1]).toBe(loss)
    expect(topo).toContain(out.y_hat)
    expect(topo).toContain(out.z_o)
    expect(topo).toContain(net.o.w1)
    expect(topo).toContain(net.h1.w1)
    expect(topo).toContain(out.x1)

    // Verify alias exports
    expect(bce).toBe(binaryCrossEntropy)
    expect(bceLoss).toBe(binaryCrossEntropy)
  })

  // 8. Numerical safety clamping
  it('safely handles boundary predictions (0 and 1) without producing NaN or Infinity loss', () => {
    const lossNearZero = binaryCrossEntropy(new Value(0), 1)
    expect(Number.isFinite(lossNearZero.data)).toBe(true)
    expect(Number.isNaN(lossNearZero.data)).toBe(false)

    const lossNearOne = binaryCrossEntropy(new Value(1), 0)
    expect(Number.isFinite(lossNearOne.data)).toBe(true)
    expect(Number.isNaN(lossNearOne.data)).toBe(false)
  })

  // 9. Regression test: extreme yHat maintains graph connectivity and non-zero gradients on backward()
  it('preserves graph connectivity and propagates gradients through original yHat even with extreme values (0 and 1)', () => {
    // Extreme 0 prediction when target is 1
    const extremeZero = new Value(0, [], '', 'extreme_zero')
    const loss0 = binaryCrossEntropy(extremeZero, 1)
    expect(topologicalOrder(loss0)).toContain(extremeZero)

    loss0.backward()
    expect(Number.isFinite(extremeZero.grad)).toBe(true)
    expect(extremeZero.grad).not.toBe(0)
    expect(extremeZero.grad).toBeLessThan(0) // dL/dyHat = -1/eps < 0

    // Extreme 1 prediction when target is 0
    const extremeOne = new Value(1, [], '', 'extreme_one')
    const loss1 = binaryCrossEntropy(extremeOne, 0)
    expect(topologicalOrder(loss1)).toContain(extremeOne)

    loss1.backward()
    expect(Number.isFinite(extremeOne.grad)).toBe(true)
    expect(extremeOne.grad).not.toBe(0)
    expect(extremeOne.grad).toBeGreaterThan(0) // dL/dyHat = 1/eps > 0

    // Extreme value coming from a upstream computation node (e.g. saturated sigmoid)
    const zExtreme = new Value(50, [], '', 'z_extreme')
    const yHatExtreme = zExtreme.sigmoid()
    const lossExtreme = binaryCrossEntropy(yHatExtreme, 0)

    lossExtreme.backward()
    expect(Number.isFinite(yHatExtreme.grad)).toBe(true)
    expect(yHatExtreme.grad).not.toBe(0)
    expect(Number.isFinite(zExtreme.grad)).toBe(true)
    expect(topologicalOrder(lossExtreme)).toContain(zExtreme)
  })
})
