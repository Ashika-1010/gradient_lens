import { describe, it, expect } from 'vitest'
import { initNetwork, forward, params } from '../../src/engine/network.js'
import { XOR_DATA } from '../../src/engine/xorData.js'

describe('Fixed Neural Network & Forward Pass (Milestone 3)', () => {
  // 1. Parameter count and labels
  it('creates exactly 9 trainable parameters with correct labels in initNetwork', () => {
    const net = initNetwork()
    const p = params(net)

    expect(p).toHaveLength(9)

    const labels = p.map((param) => param.label).sort()
    const expectedLabels = [
      'b_h1',
      'b_h2',
      'b_o',
      'w_h1_x1',
      'w_h1_x2',
      'w_h2_x1',
      'w_h2_x2',
      'w_o_h1',
      'w_o_h2',
    ].sort()

    expect(labels).toEqual(expectedLabels)
  })

  // 2. Custom weights
  it('respects custom initial weights provided to initNetwork', () => {
    const customWeights = {
      h1: { w1: 1.0, w2: 2.0, b: 3.0 },
      h2: { w1: 4.0, w2: 5.0, b: 6.0 },
      o:  { w1: 7.0, w2: 8.0, b: 9.0 },
    }

    const net = initNetwork(customWeights)

    expect(net.h1.w1.data).toBe(1.0)
    expect(net.h1.w2.data).toBe(2.0)
    expect(net.h1.b.data).toBe(3.0)

    expect(net.h2.w1.data).toBe(4.0)
    expect(net.h2.w2.data).toBe(5.0)
    expect(net.h2.b.data).toBe(6.0)

    expect(net.o.w1.data).toBe(7.0)
    expect(net.o.w2.data).toBe(8.0)
    expect(net.o.b.data).toBe(9.0)
  })

  // 3. Hand-calculated forward pass: pre-activations z_h1 and z_h2
  it('computes correct hidden pre-activations z_h1 and z_h2 for input [1, 0]', () => {
    const net = initNetwork()
    const out = forward(net, [1, 0])

    // z_h1 = 0.5 * 1 + (-0.4) * 0 + 0.1 = 0.6
    expect(out.z_h1.data).toBeCloseTo(0.6, 6)

    // z_h2 = 0.3 * 1 + 0.8 * 0 + (-0.2) = 0.1
    expect(out.z_h2.data).toBeCloseTo(0.1, 6)
  })

  // 4. Hand-calculated forward pass: hidden activations a_h1 and a_h2
  it('computes correct hidden activations a_h1 and a_h2 for input [1, 0]', () => {
    const net = initNetwork()
    const out = forward(net, [1, 0])

    // a_h1 = sigmoid(0.6) ≈ 0.645656
    expect(out.a_h1.data).toBeCloseTo(0.645656, 5)

    // a_h2 = sigmoid(0.1) ≈ 0.524979
    expect(out.a_h2.data).toBeCloseTo(0.524979, 5)
  })

  // 5. Hand-calculated forward pass: output pre-activation z_o and prediction y_hat
  it('computes correct output pre-activation z_o and prediction y_hat for input [1, 0]', () => {
    const net = initNetwork()
    const out = forward(net, [1, 0])

    // z_o = 0.7 * a_h1 + (-0.6) * a_h2 + 0.05 ≈ 0.186972
    expect(out.z_o.data).toBeCloseTo(0.186972, 5)

    // y_hat = sigmoid(z_o) ≈ 0.546604
    expect(out.y_hat.data).toBeCloseTo(0.546604, 5)
  })

  // 6. Correct labels across all intermediate and output nodes
  it('assigns correct labels to all intermediate and output nodes in the forward pass', () => {
    const net = initNetwork()
    const out = forward(net, [1, 0])

    expect(out.x1.label).toBe('x1')
    expect(out.x2.label).toBe('x2')
    expect(out.z_h1.label).toBe('z_h1')
    expect(out.a_h1.label).toBe('a_h1')
    expect(out.z_h2.label).toBe('z_h2')
    expect(out.a_h2.label).toBe('a_h2')
    expect(out.z_o.label).toBe('z_o')
    expect(out.y_hat.label).toBe('y_hat')
  })

  // 7. Graph reconstruction & parameter reuse
  it('creates a fresh computation graph on each forward call while reusing persistent parameter objects', () => {
    const net = initNetwork()
    const out1 = forward(net, [1, 0])
    const out2 = forward(net, [0, 1])

    // Graph nodes must be new instances
    expect(out1.z_h1).not.toBe(out2.z_h1)
    expect(out1.a_h1).not.toBe(out2.a_h1)
    expect(out1.z_h2).not.toBe(out2.z_h2)
    expect(out1.a_h2).not.toBe(out2.a_h2)
    expect(out1.z_o).not.toBe(out2.z_o)
    expect(out1.y_hat).not.toBe(out2.y_hat)

    // Both graphs must reuse the exact same parameter Value instances
    expect(net.h1.w1).toBe(net.h1.w1)
    // Parameter values must not be mutated by forward
    expect(net.h1.w1.data).toBe(0.5)
    expect(net.h1.w2.data).toBe(-0.4)
    expect(net.h1.b.data).toBe(0.1)
  })

  // 8. Input leaves are fresh and excluded from params(net)
  it('ensures input leaves x1 and x2 are fresh and not included in params(net)', () => {
    const net = initNetwork()
    const out = forward(net, [1, 0])
    const netParams = params(net)

    expect(netParams).not.toContain(out.x1)
    expect(netParams).not.toContain(out.x2)
    expect(out.x1.parents).toEqual([])
    expect(out.x2.parents).toEqual([])
  })

  // 9. XOR data evaluation & structural backward gradient flow
  it('evaluates all 4 XOR data samples with valid probabilities and verifies backward gradient flow', () => {
    const net = initNetwork()

    // Test all four XOR inputs
    for (const sample of XOR_DATA) {
      const out = forward(net, sample.input)
      expect(Number.isFinite(out.y_hat.data)).toBe(true)
      expect(out.y_hat.data).toBeGreaterThan(0)
      expect(out.y_hat.data).toBeLessThan(1)
    }

    // Structural backward check: all 9 parameters receive non-zero gradients when inputs are non-zero
    const out = forward(net, [1, 1])
    out.y_hat.backward()

    const netParams = params(net)
    for (const param of netParams) {
      expect(Number.isFinite(param.grad)).toBe(true)
      expect(param.grad).not.toBe(0)
    }
  })
})
