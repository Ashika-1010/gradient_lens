import { describe, it, expect } from 'vitest'
import { initNetwork, forward, params } from '../../src/engine/network.js'
import { Value } from '../../src/engine/value.js'
import { buildSnapshot } from '../../src/state/snapshot.js'

describe('State Snapshot (Milestone 9)', () => {
  // 1. Exactly 9 weights captured
  it('captures exactly 9 parameter weights in the snapshot dictionary', () => {
    const net = initNetwork()
    const snapshot = buildSnapshot(net)

    const weightKeys = Object.keys(snapshot.weights)
    expect(weightKeys).toHaveLength(9)

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
    expect(weightKeys.sort()).toEqual(expectedLabels.sort())
  })

  // 2. Values and gradients match engine parameters
  it('correctly extracts weight values and accumulated gradients from engine parameters', () => {
    const net = initNetwork()
    net.o.w1.grad = -0.2927
    net.h1.b.data = 0.777

    const snapshot = buildSnapshot(net)

    expect(snapshot.weights.w_o_h1.value).toBe(0.7)
    expect(snapshot.weights.w_o_h1.grad).toBe(-0.2927)
    expect(snapshot.weights.b_h1.value).toBe(0.777)
  })

  // 3. Activations are null when forwardOutput is omitted
  it('sets activations to null when forwardOutput is not provided', () => {
    const net = initNetwork()
    const snapshot = buildSnapshot(net)

    expect(snapshot.activations).toBeNull()
  })

  // 4. Activations match known forward-pass values when provided
  it('extracts correct activation values matching the forward pass for input [1, 0]', () => {
    const net = initNetwork()
    const out = forward(net, [1, 0])
    const snapshot = buildSnapshot(net, out)

    expect(snapshot.activations).not.toBeNull()
    expect(snapshot.activations.x1).toBe(1)
    expect(snapshot.activations.x2).toBe(0)
    expect(snapshot.activations.z_h1).toBeCloseTo(0.6, 6)
    expect(snapshot.activations.a_h1).toBeCloseTo(0.645656, 5)
    expect(snapshot.activations.z_h2).toBeCloseTo(0.1, 6)
    expect(snapshot.activations.a_h2).toBeCloseTo(0.524979, 5)
    expect(snapshot.activations.z_o).toBeCloseTo(0.186972, 5)
    expect(snapshot.activations.y_hat).toBeCloseTo(0.546604, 5)
  })

  // 5. Every snapshot property is a primitive number (no Value objects)
  it('guarantees that all weights and activations are stored purely as primitive numbers', () => {
    const net = initNetwork()
    const out = forward(net, [1, 0])
    const snapshot = buildSnapshot(net, out)

    for (const key of Object.keys(snapshot.weights)) {
      const entry = snapshot.weights[key]
      expect(typeof entry.value).toBe('number')
      expect(typeof entry.grad).toBe('number')
      expect(entry.value).not.toBeInstanceOf(Value)
      expect(entry.grad).not.toBeInstanceOf(Value)
    }

    for (const key of Object.keys(snapshot.activations)) {
      const val = snapshot.activations[key]
      expect(typeof val).toBe('number')
      expect(val).not.toBeInstanceOf(Value)
    }
  })

  // 6. Snapshot immutability: engine mutation does not affect existing snapshot
  it('maintains independence so subsequent engine mutations do not corrupt previously built snapshots', () => {
    const net = initNetwork()
    const out = forward(net, [1, 0])
    const snapshot = buildSnapshot(net, out)

    // Mutate engine nodes
    net.o.w1.data = 999.0
    net.o.w1.grad = 888.0
    out.y_hat.data = 0.123

    // Snapshot retains its historical captured values
    expect(snapshot.weights.w_o_h1.value).toBe(0.7)
    expect(snapshot.weights.w_o_h1.grad).toBe(0)
    expect(snapshot.activations.y_hat).toBeCloseTo(0.546604, 5)
  })
})
