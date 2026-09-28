import { describe, it, expect } from 'vitest'
import { Value } from '../../src/engine/value.js'

const leaf = (x, label = '') => new Value(x, [], '', label)
const sigmoidNum = (z) => 1 / (1 + Math.exp(-z))

describe('Scalar Autodiff Operations (Milestone 2)', () => {
  describe('neg()', () => {
    // 1. forward: negating 3 gives -3
    it('negates input correctly in forward pass', () => {
      const a = leaf(3, 'a')
      const b = a.neg()
      expect(b.data).toBe(-3)
      expect(b.op).toBe('neg')
      expect(b.parents).toEqual([a])
    })

    // 2. backward: gradient is -1
    it('computes gradient of negation as -1', () => {
      const a = leaf(3, 'a')
      const b = a.neg()
      b.backward()
      expect(b.grad).toBe(1.0)
      expect(a.grad).toBe(-1.0)
    })
  })

  describe('sub()', () => {
    // 3. 5 - 3 = 2
    it('computes forward subtraction correctly', () => {
      const a = leaf(5, 'a')
      const b = leaf(3, 'b')
      const c = a.sub(b)
      expect(c.data).toBe(2)
      expect(c.op).toBe('+')
    })

    // 4. left gradient is +1, right gradient is -1
    it('computes correct backward gradients for subtraction (+1 and -1)', () => {
      const a = leaf(5, 'a')
      const b = leaf(3, 'b')
      const c = a.sub(b)
      c.backward()
      expect(c.grad).toBe(1.0)
      expect(a.grad).toBe(1.0)
      expect(b.grad).toBe(-1.0)
    })

    // 5. a - a has gradient 0
    it('computes gradient of a - a as 0', () => {
      const a = leaf(5, 'a')
      const c = a.sub(a)
      expect(c.data).toBe(0)
      c.backward()
      expect(a.grad).toBe(0)
    })

    // 6. Value.from(1).sub(p) works and gives p.grad = -1
    it('supports 1 - p pattern used in binary cross-entropy', () => {
      const p = leaf(0.7, 'p')
      const oneMinusP = Value.from(1).sub(p)
      expect(oneMinusP.data).toBeCloseTo(0.3)
      oneMinusP.backward()
      expect(p.grad).toBe(-1.0)
    })
  })

  describe('sigmoid()', () => {
    // 7. sigmoid(0) = 0.5, sigmoid(2) ≈ 0.8807970779778823, sigmoid(-2) ≈ 0.11920292202211755
    it('computes forward sigmoid values accurately', () => {
      const z0 = leaf(0, 'z0').sigmoid()
      expect(z0.data).toBe(0.5)

      const z2 = leaf(2, 'z2').sigmoid()
      expect(z2.data).toBeCloseTo(0.8807970779778823, 10)

      const zNeg2 = leaf(-2, 'zNeg2').sigmoid()
      expect(zNeg2.data).toBeCloseTo(0.11920292202211755, 10)
    })

    // 8. symmetry: sigmoid(-z) = 1 - sigmoid(z)
    it('satisfies the sigmoid symmetry property sigmoid(-z) = 1 - sigmoid(z)', () => {
      const z = 1.75
      const sPos = leaf(z).sigmoid()
      const sNeg = leaf(-z).sigmoid()
      expect(sNeg.data).toBeCloseTo(1 - sPos.data, 10)
    })

    // 9. derivative at 0 = 0.25
    it('computes derivative of sigmoid at 0 as 0.25', () => {
      const z = leaf(0, 'z')
      const s = z.sigmoid()
      s.backward()
      expect(z.grad).toBeCloseTo(0.25, 10)
    })

    // 10. derivative at ±2 ≈ 0.1049935854
    it('computes derivative of sigmoid at ±2 accurately', () => {
      const zPos = leaf(2, 'zPos')
      const sPos = zPos.sigmoid()
      sPos.backward()
      expect(zPos.grad).toBeCloseTo(0.1049935854, 8)

      const zNeg = leaf(-2, 'zNeg')
      const sNeg = zNeg.sigmoid()
      sNeg.backward()
      expect(zNeg.grad).toBeCloseTo(0.1049935854, 8)
    })

    // 11. extreme inputs ±50 and ±1000 produce finite output and finite gradient
    it('handles extreme inputs ±50 and ±1000 producing finite outputs and gradients', () => {
      for (const val of [50, -50, 1000, -1000]) {
        const z = leaf(val)
        const s = z.sigmoid()
        expect(Number.isFinite(s.data)).toBe(true)
        s.backward()
        expect(Number.isFinite(z.grad)).toBe(true)
      }
    })
  })

  describe('log()', () => {
    // 12. log(1) = 0, log(2) ≈ Math.LN2, log(Math.E) ≈ 1
    it('computes natural logarithm accurately in forward pass', () => {
      const l1 = leaf(1).log()
      expect(l1.data).toBe(0)

      const l2 = leaf(2).log()
      expect(l2.data).toBeCloseTo(Math.LN2, 10)

      const lE = leaf(Math.E).log()
      expect(lE.data).toBeCloseTo(1.0, 10)
    })

    // 13. derivative at 2 = 0.5, derivative at 0.5 = 2
    it('computes gradient of log as 1/x', () => {
      const x1 = leaf(2, 'x1')
      x1.log().backward()
      expect(x1.grad).toBeCloseTo(0.5, 10)

      const x2 = leaf(0.5, 'x2')
      x2.log().backward()
      expect(x2.grad).toBeCloseTo(2.0, 10)
    })

    // 14. log(0) = -Infinity
    it('produces -Infinity for log(0) without clamping', () => {
      const zero = leaf(0, 'zero')
      const lZero = zero.log()
      expect(lZero.data).toBe(-Infinity)
    })
  })

  describe('Composition & Chain Rule', () => {
    // 15. Single neuron: x = 1.5, w = -2, b = 0.5, a = x.mul(w).add(b).sigmoid()
    it('computes forward pass and gradients for a single sigmoid neuron', () => {
      const x = leaf(1.5, 'x')
      const w = leaf(-2, 'w')
      const b = leaf(0.5, 'b')

      const z = x.mul(w).add(b)
      const a = z.sigmoid()

      // Forward checks
      expect(z.data).toBe(-2.5)
      expect(a.data).toBeCloseTo(sigmoidNum(-2.5), 10)

      a.backward()

      // Gradient checks: sigma'(-2.5) = sigma(-2.5) * (1 - sigma(-2.5))
      const sigPrime = sigmoidNum(-2.5) * (1 - sigmoidNum(-2.5))
      expect(w.grad).toBeCloseTo(sigPrime * 1.5, 10)
      expect(x.grad).toBeCloseTo(sigPrime * -2, 10)
      expect(b.grad).toBeCloseTo(sigPrime, 10)
    })

    // 16. Reuse through nonlinearity: s = sigmoid(a), s * s => da = 2 * s * s * (1 - s)
    it('computes gradients when a sigmoid output is reused in multiplication (s * s)', () => {
      const a = leaf(1.2, 'a')
      const s = a.sigmoid()
      const out = s.mul(s)

      out.backward()

      const sVal = sigmoidNum(1.2)
      // d(s^2)/da = 2 * s * s' = 2 * s * s * (1 - s)
      const expectedGrad = 2 * sVal * sVal * (1 - sVal)
      expect(a.grad).toBeCloseTo(expectedGrad, 10)
    })

    // 17. Finite-difference check: autodiff vs central finite differences for log(sigmoid(w*x + b))
    it('matches independent central finite-difference gradient for log(sigmoid(w*x + b))', () => {
      const wInit = 0.8
      const xInit = -1.2
      const bInit = 0.3
      const eps = 1e-6

      // Independent pure numeric function
      const lossFn = (wVal, xVal, bVal) => Math.log(sigmoidNum(wVal * xVal + bVal))

      const numGradW = (lossFn(wInit + eps, xInit, bInit) - lossFn(wInit - eps, xInit, bInit)) / (2 * eps)
      const numGradX = (lossFn(wInit, xInit + eps, bInit) - lossFn(wInit, xInit - eps, bInit)) / (2 * eps)
      const numGradB = (lossFn(wInit, xInit, bInit + eps) - lossFn(wInit, xInit, bInit - eps)) / (2 * eps)

      // Value autodiff graph
      const w = leaf(wInit, 'w')
      const x = leaf(xInit, 'x')
      const b = leaf(bInit, 'b')
      const out = w.mul(x).add(b).sigmoid().log()

      out.backward()

      expect(w.grad).toBeCloseTo(numGradW, 5)
      expect(x.grad).toBeCloseTo(numGradX, 5)
      expect(b.grad).toBeCloseTo(numGradB, 5)
    })
  })
})
