import { describe, it, expect } from 'vitest'
import { initNetwork, forward, params } from '../../src/engine/network.js'
import { bceLoss } from '../../src/engine/loss.js'
import { XOR_DATA } from '../../src/engine/xorData.js'
import { PARAM_META, getParamNode, buildChainExpression } from '../../src/engine/chainRule.js'

describe('Chain-Rule Calculations & Inspector Data (Milestone 12)', () => {
  const ALL_PARAM_LABELS = [
    'w_o_h1',
    'w_o_h2',
    'b_o',
    'w_h1_x1',
    'w_h1_x2',
    'b_h1',
    'w_h2_x1',
    'w_h2_x2',
    'b_h2',
  ]

  // 1. PARAM_META coverage
  describe('PARAM_META', () => {
    it('covers all nine parameters with valid labels and formulas', () => {
      expect(Object.keys(PARAM_META)).toHaveLength(9)

      for (const label of ALL_PARAM_LABELS) {
        const meta = PARAM_META[label]
        expect(meta).toBeDefined()
        expect(meta.label).toBe(label)
        expect(typeof meta.formula).toBe('string')
        expect(meta.formula.length).toBeGreaterThan(0)
        expect(['output', 'hidden']).toContain(meta.layer)
        expect(['weight', 'bias']).toContain(meta.type)
      }
    })
  })

  // 2. buildChainExpression on default XOR example [1, 0], y = 1
  describe('buildChainExpression with [1, 0], y = 1', () => {
    it('accurately computes chain-rule factors agreeing with autodiff engine within 1e-6 for all 9 parameters', () => {
      const net = initNetwork()
      const input = [1, 0]
      const y = 1
      const out = forward(net, input)
      const loss = bceLoss(out.y_hat, y)
      loss.backward()

      for (const label of ALL_PARAM_LABELS) {
        const expr = buildChainExpression(net, out, label, y)
        expect(expr).not.toBeNull()
        expect(expr.label).toBe(label)
        expect(typeof expr.formula).toBe('string')
        expect(Array.isArray(expr.factors)).toBe(true)
        expect(expr.factors.length).toBeGreaterThan(0)

        // Verify product agrees with engine gradient within 1e-6
        expect(expr.engineGrad).toBeDefined()
        expect(Math.abs(expr.product - expr.engineGrad)).toBeLessThan(1e-6)
        expect(expr.agrees).toBe(true)
      }
    })

    it('correctly produces zero gradients for w_h1_x2 and w_h2_x2 when x2 = 0', () => {
      const net = initNetwork()
      const input = [1, 0]
      const y = 1
      const out = forward(net, input)
      const loss = bceLoss(out.y_hat, y)
      loss.backward()

      const expr_h1_x2 = buildChainExpression(net, out, 'w_h1_x2', y)
      expect(expr_h1_x2.product).toBe(0)
      expect(expr_h1_x2.engineGrad).toBe(0)
      expect(expr_h1_x2.agrees).toBe(true)

      const expr_h2_x2 = buildChainExpression(net, out, 'w_h2_x2', y)
      expect(expr_h2_x2.product).toBe(0)
      expect(expr_h2_x2.engineGrad).toBe(0)
      expect(expr_h2_x2.agrees).toBe(true)
    })
  })

  // 3. Independent validation across all 4 XOR samples
  describe('All XOR samples cross-check', () => {
    it('agrees with engine gradients within 1e-6 across all 4 XOR samples for all 9 parameters', () => {
      for (const sample of XOR_DATA) {
        const net = initNetwork()
        const out = forward(net, sample.input)
        const loss = bceLoss(out.y_hat, sample.label)
        loss.backward()

        for (const label of ALL_PARAM_LABELS) {
          const expr = buildChainExpression(net, out, label, sample.label)
          expect(expr).not.toBeNull()
          expect(Math.abs(expr.product - expr.engineGrad)).toBeLessThan(1e-6)
          expect(expr.agrees).toBe(true)
        }
      }
    })
  })

  // 4. Immutability: does not mutate engine nodes
  describe('Immutability and purity', () => {
    it('does not mutate parameter values or engine gradients', () => {
      const net = initNetwork()
      const out = forward(net, [1, 0])
      const loss = bceLoss(out.y_hat, 1)
      loss.backward()

      // Record baseline values and grads
      const baseline = params(net).map((p) => ({ label: p.label, data: p.data, grad: p.grad }))

      // Call buildChainExpression multiple times
      for (const label of ALL_PARAM_LABELS) {
        buildChainExpression(net, out, label, 1)
      }

      // Verify no changes occurred
      params(net).forEach((p, i) => {
        expect(p.data).toBe(baseline[i].data)
        expect(p.grad).toBe(baseline[i].grad)
      })
    })

    it('returns fully JSON-serializable plain object data', () => {
      const net = initNetwork()
      const out = forward(net, [1, 0])
      const loss = bceLoss(out.y_hat, 1)
      loss.backward()

      const expr = buildChainExpression(net, out, 'w_o_h1', 1)
      const serialized = JSON.stringify(expr)
      expect(typeof serialized).toBe('string')
      const parsed = JSON.parse(serialized)

      expect(parsed.label).toBe('w_o_h1')
      expect(parsed.formula).toBe('(ŷ - y) * a_h1')
      expect(parsed.product).toBeCloseTo(-0.2927, 4)
      expect(parsed.engineGrad).toBeCloseTo(-0.2927, 4)
    })

    it('returns null gracefully on unknown label or invalid arguments', () => {
      const net = initNetwork()
      const out = forward(net, [1, 0])

      expect(buildChainExpression(net, out, 'unknown_param', 1)).toBeNull()
      expect(buildChainExpression(null, out, 'w_o_h1', 1)).toBeNull()
      expect(buildChainExpression(net, null, 'w_o_h1', 1)).toBeNull()
    })
  })

  // 5. Mismatch detection test
  describe('Mismatch detection verification', () => {
    it('detects a mismatch if the wrong output weight is selected for a hidden neuron', () => {
      const net = initNetwork()
      const out = forward(net, [1, 0])
      const loss = bceLoss(out.y_hat, 1)
      loss.backward()

      // For w_h2_x1, the correct downstream output weight is w_o_h2 (net.o.w2).
      // If someone erroneously uses w_o_h1 (net.o.w1) instead:
      const dL_dzo = out.y_hat.data - 1
      const sigPrime_h2 = out.a_h2.data * (1 - out.a_h2.data)
      const x1 = out.x1.data

      // Wrong calculation using w_o_h1 (0.7) instead of w_o_h2 (-0.6)
      const wrongProduct = dL_dzo * net.o.w1.data * sigPrime_h2 * x1
      const realEngineGrad = net.h2.w1.grad // uses w_o_h2

      const mismatchDiff = Math.abs(wrongProduct - realEngineGrad)
      // Mismatch is large and detectable (> 0.1)
      expect(mismatchDiff).toBeGreaterThan(0.1)
      expect(mismatchDiff < 1e-6).toBe(false)

      // The correct implementation in buildChainExpression agrees within 1e-6
      const correctExpr = buildChainExpression(net, out, 'w_h2_x1', 1)
      expect(correctExpr.agrees).toBe(true)
      expect(Math.abs(correctExpr.product - correctExpr.engineGrad)).toBeLessThan(1e-6)
    })
  })
})
