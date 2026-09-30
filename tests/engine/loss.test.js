import { describe, it, expect } from 'vitest'
import { Value, topologicalOrder } from '../../src/engine/value.js'
import { bceLoss, binaryCrossEntropy, EPS } from '../../src/engine/loss.js'

const leaf = (v, label = '') => new Value(v, [], '', label)

describe('Value.clamp() & Binary Cross-Entropy Loss (Milestone 4)', () => {
  describe('Value.clamp()', () => {
    // 1. Forward passes values inside unchanged
    it('passes values strictly inside the range unchanged in forward pass', () => {
      const a = leaf(0.5, 'a')
      const c = a.clamp(0.1, 0.9)
      expect(c.data).toBe(0.5)
      expect(c.op).toBe('clamp')
      expect(c.parents).toEqual([a])
    })

    // 2. Forward clips values above the maximum
    it('clips values strictly above the maximum to the maximum bound', () => {
      const a = leaf(1.5, 'a')
      const c = a.clamp(0.1, 0.9)
      expect(c.data).toBe(0.9)
    })

    // 3. Forward clips values below the minimum
    it('clips values strictly below the minimum to the minimum bound', () => {
      const a = leaf(-0.5, 'a')
      const c = a.clamp(0.1, 0.9)
      expect(c.data).toBe(0.1)
    })

    // 4. Backward passes gradient unchanged when inside
    it('passes gradient unchanged when inside the range (min < data < max)', () => {
      const a = leaf(0.5, 'a')
      const c = a.clamp(0.1, 0.9)
      c.backward()
      expect(c.grad).toBe(1.0)
      expect(a.grad).toBe(1.0)
    })

    // 5. Backward blocks gradient when clamped (at or outside boundaries)
    it('blocks gradient when clamped outside or on the boundaries', () => {
      // Outside above max
      const aAbove = leaf(5, 'aAbove')
      aAbove.clamp(0.1, 0.9).backward()
      expect(aAbove.grad).toBe(0)

      // Outside below min
      const aBelow = leaf(-5, 'aBelow')
      aBelow.clamp(0.1, 0.9).backward()
      expect(aBelow.grad).toBe(0)

      // Exactly at min boundary (strict inequality blocks gradient)
      const aMin = leaf(0.1, 'aMin')
      aMin.clamp(0.1, 0.9).backward()
      expect(aMin.grad).toBe(0)

      // Exactly at max boundary (strict inequality blocks gradient)
      const aMax = leaf(0.9, 'aMax')
      aMax.clamp(0.1, 0.9).backward()
      expect(aMax.grad).toBe(0)
    })
  })

  describe('bceLoss()', () => {
    // 6. yHat ≈ 1, y = 1 → near-zero loss
    it('produces near-zero loss when yHat ≈ 1 and y = 1', () => {
      const yHat = leaf(0.9999, 'y_hat')
      const loss = bceLoss(yHat, 1)
      expect(loss.data).toBeCloseTo(0.0001, 3)
      expect(loss.data).toBeGreaterThan(0)
    })

    // 7. yHat ≈ 1, y = 0 → large loss
    it('produces large loss when yHat ≈ 1 and y = 0', () => {
      const yHat = leaf(0.9999, 'y_hat')
      const loss = bceLoss(yHat, 0)
      expect(loss.data).toBeGreaterThan(9.0)
    })

    // 8. yHat = 0.5 → ln(2)
    it('computes loss as ln(2) when yHat = 0.5 for both y = 1 and y = 0', () => {
      const loss1 = bceLoss(leaf(0.5), 1)
      expect(loss1.data).toBeCloseTo(Math.LN2, 10)

      const loss0 = bceLoss(leaf(0.5), 0)
      expect(loss0.data).toBeCloseTo(Math.LN2, 10)
    })

    // 9. Hand fixture: yHat ≈ 0.546604, y = 1, loss ≈ 0.6040
    it('computes loss ≈ 0.6040 for the hand-calculated fixture yHat = 0.546604, y = 1', () => {
      const yHat = leaf(0.546604, 'y_hat')
      const loss = bceLoss(yHat, 1)
      expect(loss.data).toBeCloseTo(0.60403, 4)
      expect(loss.label).toBe('L')
    })

    // 10. Numerical safety at extreme values [0, 1], [1, 0], [0, 0], [1, 1]
    it('maintains finite loss and finite backward gradients for extreme predictions [0, 1]', () => {
      const pairs = [
        { yHat: 0, y: 1 },
        { yHat: 1, y: 0 },
        { yHat: 0, y: 0 },
        { yHat: 1, y: 1 },
      ]

      for (const { yHat: yHatVal, y: yVal } of pairs) {
        const yHatNode = leaf(yHatVal, `yHat_${yHatVal}`)
        const loss = bceLoss(yHatNode, yVal)

        expect(Number.isFinite(loss.data)).toBe(true)
        expect(Number.isNaN(loss.data)).toBe(false)

        loss.backward()
        expect(Number.isFinite(yHatNode.grad)).toBe(true)
        expect(Number.isNaN(yHatNode.grad)).toBe(false)
      }
    })

    // 11. Gradient correctness: dL/dyHat = -y/yHat + (1-y)/(1-yHat)
    it('computes exact analytical gradient dL/dyHat via the autodiff graph', () => {
      const yHat = leaf(0.7, 'y_hat')
      const y = 1
      const loss = bceLoss(yHat, y)

      loss.backward()

      // Analytical: -1 / 0.7 ≈ -1.4285714
      const expectedGrad = -y / 0.7 + (1 - y) / (1 - 0.7)
      expect(yHat.grad).toBeCloseTo(expectedGrad, 6)

      // When y = 0
      const yHat0 = leaf(0.3, 'y_hat0')
      const loss0 = bceLoss(yHat0, 0)
      loss0.backward()
      // Analytical: 1 / (1 - 0.3) = 1 / 0.7 ≈ 1.4285714
      expect(yHat0.grad).toBeCloseTo(1 / 0.7, 6)
    })

    // 12. Graph connectivity: yHat -> clamp -> log -> BCE loss
    it('preserves full graph connectivity with yHat as an ancestor of L', () => {
      const yHat = leaf(0.65, 'y_hat')
      const loss = bceLoss(yHat, 1)

      expect(loss).toBeInstanceOf(Value)
      expect(loss.label).toBe('L')

      const topo = topologicalOrder(loss)
      expect(topo[topo.length - 1]).toBe(loss)
      expect(topo).toContain(yHat)

      // Find clamp node in graph
      const clampNode = topo.find((node) => node.op === 'clamp')
      expect(clampNode).toBeDefined()
      expect(clampNode.parents).toContain(yHat)

      // Check alias export
      expect(binaryCrossEntropy).toBe(bceLoss)
    })
  })
})
