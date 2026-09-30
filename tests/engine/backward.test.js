import { describe, it, expect } from 'vitest'
import { initNetwork, forward, params } from '../../src/engine/network.js'
import { bceLoss } from '../../src/engine/loss.js'
import { XOR_DATA } from '../../src/engine/xorData.js'

describe('End-to-End Backward Propagation Pipeline (Milestone 4)', () => {
  const sigmoid = (z) => 1 / (1 + Math.exp(-z))

  // 1 & 2 & 3. Complete pipeline test with independent hand calculations for [1, 0], y = 1
  it('computes correct forward loss and backpropagates gradients matching independent chain rule', () => {
    const net = initNetwork()
    const out = forward(net, [1, 0])
    const loss = bceLoss(out.y_hat, 1)

    // 1. Loss ≈ 0.6040
    expect(loss.data).toBeCloseTo(0.6040, 3)

    loss.backward()

    // 2. Output weight gradient: net.o.w1.grad ≈ -0.2927
    expect(net.o.w1.grad).toBeCloseTo(-0.2927, 4)

    // 3. Independent plain-number hand calculation for hidden weight w_h1_x1:
    // x1 = 1, x2 = 0
    const x1 = 1.0
    const x2 = 0.0
    const w_h1_x1 = 0.5
    const w_h1_x2 = -0.4
    const b_h1 = 0.1
    const z_h1 = w_h1_x1 * x1 + w_h1_x2 * x2 + b_h1 // 0.6
    const a_h1 = sigmoid(z_h1)

    const w_h2_x1 = 0.3
    const w_h2_x2 = 0.8
    const b_h2 = -0.2
    const z_h2 = w_h2_x1 * x1 + w_h2_x2 * x2 + b_h2 // 0.1
    const a_h2 = sigmoid(z_h2)

    const w_o_h1 = 0.7
    const w_o_h2 = -0.6
    const b_o = 0.05
    const z_o = w_o_h1 * a_h1 + w_o_h2 * a_h2 + b_o
    const y_hat = sigmoid(z_o)

    const y = 1.0
    const dL_dz_o = y_hat - y // ≈ -0.453396
    const sigPrime_z_h1 = a_h1 * (1.0 - a_h1) // sigmoid'(z_h1)

    // Independent analytical derivative: dL/dw_h1_x1 = (dL/dz_o) * w_o_h1 * sigmoid'(z_h1) * x1
    const expected_grad_w_h1_x1 = dL_dz_o * w_o_h1 * sigPrime_z_h1 * x1

    expect(net.h1.w1.grad).toBeCloseTo(expected_grad_w_h1_x1, 6)
    expect(net.h1.w1.grad).toBeCloseTo(-0.0726, 3)
  })

  // 4. Non-zero gradients across network parameters
  it('propagates non-zero gradients to all active parameters for input [1, 0] and all 9 parameters for [1, 1]', () => {
    // For [1, 0], the 7 active parameters receive non-zero gradients
    const net10 = initNetwork()
    const out10 = forward(net10, [1, 0])
    const loss10 = bceLoss(out10.y_hat, 1)
    loss10.backward()

    expect(net10.h1.w1.grad).not.toBe(0)
    expect(net10.h1.b.grad).not.toBe(0)
    expect(net10.h2.w1.grad).not.toBe(0)
    expect(net10.h2.b.grad).not.toBe(0)
    expect(net10.o.w1.grad).not.toBe(0)
    expect(net10.o.w2.grad).not.toBe(0)
    expect(net10.o.b.grad).not.toBe(0)

    // For [1, 1] where both inputs are non-zero, every single one of the 9 parameters receives non-zero gradient
    const net11 = initNetwork()
    const out11 = forward(net11, [1, 1])
    const loss11 = bceLoss(out11.y_hat, 0)
    loss11.backward()

    for (const p of params(net11)) {
      expect(Number.isFinite(p.grad)).toBe(true)
      expect(p.grad).not.toBe(0)
    }
  })

  // 5. Changing label from correct to wrong case produces larger-magnitude gradient
  it('produces a larger-magnitude output gradient when target label is wrong (y = 0) vs aligned (y = 1)', () => {
    // For [1, 0], y_hat ≈ 0.5466.
    // When y = 1 (target is closer to prediction): |y_hat - 1| ≈ 0.4534
    const netCorrect = initNetwork()
    const outCorrect = forward(netCorrect, [1, 0])
    const lossCorrect = bceLoss(outCorrect.y_hat, 1)
    lossCorrect.backward()

    // When y = 0 (target is further from prediction): |y_hat - 0| ≈ 0.5466
    const netWrong = initNetwork()
    const outWrong = forward(netWrong, [1, 0])
    const lossWrong = bceLoss(outWrong.y_hat, 0)
    lossWrong.backward()

    expect(Math.abs(outWrong.z_o.grad)).toBeGreaterThan(Math.abs(outCorrect.z_o.grad))
    expect(Math.abs(netWrong.o.w1.grad)).toBeGreaterThan(Math.abs(netCorrect.o.w1.grad))
    expect(lossWrong.data).toBeGreaterThan(lossCorrect.data)
  })

  // 6. Complete XOR dataset pipeline validation
  it('runs the full forward -> bceLoss -> backward pipeline on all 4 XOR samples without error', () => {
    for (const sample of XOR_DATA) {
      const net = initNetwork()
      const out = forward(net, sample.input)
      const loss = bceLoss(out.y_hat, sample.label)

      expect(Number.isFinite(loss.data)).toBe(true)
      expect(loss.data).toBeGreaterThan(0)

      loss.backward()

      for (const p of params(net)) {
        expect(Number.isFinite(p.grad)).toBe(true)
        expect(Number.isNaN(p.grad)).toBe(false)
      }
    }
  })
})
