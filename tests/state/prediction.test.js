import { describe, it, expect } from 'vitest'
import {
  PREDICTABLE_LABELS,
  WORKED_EXAMPLE_LABEL,
  UPDATE_DIRECTIONS,
  correctDirection,
  isCorrect,
  isPredictionRequired,
  isGradientRevealed,
  recordPrediction,
  gatePredictions,
  inspectorStatus,
} from '../../src/state/prediction.js'
import { EDGES } from '../../src/visualization/layout.js'
import { initNetwork, forward, params } from '../../src/engine/network.js'
import { bceLoss } from '../../src/engine/loss.js'
import { sgdStep } from '../../src/engine/optimizer.js'

describe('Prediction Logic & Gating (Milestone 13)', () => {
  // 1. Predictable labels exactly match edge labels
  it('defines PREDICTABLE_LABELS that exactly match the six edge weight labels', () => {
    const expectedEdgeLabels = EDGES.map((e) => e.weightLabel)
    expect(PREDICTABLE_LABELS).toHaveLength(6)
    expect([...PREDICTABLE_LABELS]).toEqual(expectedEdgeLabels)

    // Ensure biases are NOT in PREDICTABLE_LABELS
    expect(PREDICTABLE_LABELS).not.toContain('b_o')
    expect(PREDICTABLE_LABELS).not.toContain('b_h1')
    expect(PREDICTABLE_LABELS).not.toContain('b_h2')
  })

  it('defines WORKED_EXAMPLE_LABEL as w_o_h1', () => {
    expect(WORKED_EXAMPLE_LABEL).toBe('w_o_h1')
  })

  it('defines UPDATE_DIRECTIONS as increase, decrease, same', () => {
    expect(UPDATE_DIRECTIONS).toEqual(['increase', 'decrease', 'same'])
  })

  // 2. Correct directions for positive, negative, zero, negative zero, and tiny gradients
  describe('correctDirection calculation', () => {
    it('returns decrease for positive gradients', () => {
      expect(correctDirection(0.5)).toBe('decrease')
      expect(correctDirection(10.0)).toBe('decrease')
      expect(correctDirection(1e-4)).toBe('decrease')
    })

    it('returns increase for negative gradients', () => {
      expect(correctDirection(-0.5)).toBe('increase')
      expect(correctDirection(-10.0)).toBe('increase')
      expect(correctDirection(-1e-4)).toBe('increase')
    })

    it('returns same for exact zero and signed negative zero', () => {
      expect(correctDirection(0)).toBe('same')
      expect(correctDirection(+0)).toBe('same')
      expect(correctDirection(-0)).toBe('same')
    })

    it('returns same for tiny gradients within tolerance 1e-9', () => {
      expect(correctDirection(1e-10)).toBe('same')
      expect(correctDirection(-1e-10)).toBe('same')
      expect(correctDirection(1e-9)).toBe('same')
      expect(correctDirection(-1e-9)).toBe('same')
    })

    it('returns correct direction for values just outside tolerance', () => {
      expect(correctDirection(1.000001e-9)).toBe('decrease')
      expect(correctDirection(-1.000001e-9)).toBe('increase')
    })

    it('returns null for non-number or NaN inputs', () => {
      expect(correctDirection(NaN)).toBeNull()
      expect(correctDirection(null)).toBeNull()
      expect(correctDirection(undefined)).toBeNull()
    })
  })

  // 3. Direction predictions agree with actual sgdStep changes across all nine parameters
  describe('Agreement with sgdStep across all 9 parameters', () => {
    it('agrees with actual sgdStep weight updates for all parameters', () => {
      const net = initNetwork()
      const out = forward(net, [1, 0])
      const loss = bceLoss(out.y_hat, 1)
      loss.backward()

      const lr = 0.05
      const beforeValues = params(net).map((p) => ({ label: p.label, val: p.data, grad: p.grad }))

      // Execute real SGD step
      sgdStep(net, lr)

      for (let i = 0; i < beforeValues.length; i++) {
        const { label, val: beforeVal, grad } = beforeValues[i]
        const afterVal = params(net)[i].data
        const predictedDir = correctDirection(grad)

        if (predictedDir === 'decrease') {
          expect(afterVal).toBeLessThan(beforeVal)
        } else if (predictedDir === 'increase') {
          expect(afterVal).toBeGreaterThan(beforeVal)
        } else if (predictedDir === 'same') {
          expect(afterVal).toBeCloseTo(beforeVal, 9)
        }
      }
    })
  })

  // 4. Worked-example behavior and bias exclusions
  describe('isPredictionRequired', () => {
    it('returns false for the worked example w_o_h1', () => {
      expect(isPredictionRequired(WORKED_EXAMPLE_LABEL)).toBe(false)
      expect(isPredictionRequired('w_o_h1')).toBe(false)
    })

    it('returns true for all other 5 predictable edge weights', () => {
      const otherWeights = PREDICTABLE_LABELS.filter((l) => l !== WORKED_EXAMPLE_LABEL)
      expect(otherWeights).toHaveLength(5)
      for (const w of otherWeights) {
        expect(isPredictionRequired(w)).toBe(true)
      }
    })

    it('returns false for biases and unknown labels', () => {
      expect(isPredictionRequired('b_o')).toBe(false)
      expect(isPredictionRequired('b_h1')).toBe(false)
      expect(isPredictionRequired('b_h2')).toBe(false)
      expect(isPredictionRequired('unknown')).toBe(false)
    })
  })

  // 5. Immutable, valid, locked predictions
  describe('recordPrediction', () => {
    it('immutably records a valid prediction', () => {
      const initial = {}
      const updated = recordPrediction(initial, 'w_o_h2', 'decrease')

      expect(initial).toEqual({})
      expect(updated).toEqual({ w_o_h2: 'decrease' })
    })

    it('locks guess once submitted and does not allow overwriting', () => {
      const initial = { w_o_h2: 'decrease' }
      const attemptedChange = recordPrediction(initial, 'w_o_h2', 'increase')

      expect(attemptedChange.w_o_h2).toBe('decrease')
      expect(attemptedChange).toBe(initial)
    })

    it('rejects invalid directions', () => {
      const initial = {}
      const res = recordPrediction(initial, 'w_o_h2', 'up')
      expect(res).toBe(initial)
    })

    it('rejects labels that do not require prediction (worked example and biases)', () => {
      const initial = {}
      expect(recordPrediction(initial, 'w_o_h1', 'decrease')).toBe(initial)
      expect(recordPrediction(initial, 'b_o', 'decrease')).toBe(initial)
    })
  })

  // 6. Two-gate behavior and rule that stage-hidden gradient can never become visible
  describe('isGradientRevealed and gatePredictions', () => {
    it('never reveals a gradient when stageRevealed is false, even if a prediction exists', () => {
      const predictions = { w_o_h2: 'decrease', w_h1_x1: 'increase' }

      expect(isGradientRevealed('w_o_h2', false, predictions)).toBe(false)
      expect(isGradientRevealed('w_h1_x1', false, predictions)).toBe(false)
      expect(isGradientRevealed('w_o_h1', false, predictions)).toBe(false)
      expect(isGradientRevealed('b_o', false, predictions)).toBe(false)
    })

    it('reveals worked example w_o_h1 and biases as soon as stageRevealed is true without prediction', () => {
      expect(isGradientRevealed('w_o_h1', true, {})).toBe(true)
      expect(isGradientRevealed('b_o', true, {})).toBe(true)
      expect(isGradientRevealed('b_h1', true, {})).toBe(true)
    })

    it('hides predictable weight when stageRevealed is true but no prediction is submitted', () => {
      expect(isGradientRevealed('w_o_h2', true, {})).toBe(false)
      expect(isGradientRevealed('w_h1_x1', true, {})).toBe(false)
    })

    it('reveals predictable weight when stageRevealed is true and prediction is submitted', () => {
      expect(isGradientRevealed('w_o_h2', true, { w_o_h2: 'decrease' })).toBe(true)
      expect(isGradientRevealed('w_h1_x1', true, { w_h1_x1: 'same' })).toBe(true)
    })

    it('gatePredictions leaves stage-hidden gradients as null and filters unpredicted gradients', () => {
      const stageWeights = {
        w_o_h1: { value: 0.7, grad: -0.2927 }, // stage revealed, worked example
        w_o_h2: { value: -0.6, grad: -0.2380 }, // stage revealed, predictable
        b_o: { value: 0.05, grad: -0.4534 }, // stage revealed, bias
        w_h1_x1: { value: 0.5, grad: null }, // stage hidden
      }

      // 1. Without predictions
      const gatedNoPred = gatePredictions(stageWeights, {})
      expect(gatedNoPred.w_o_h1.grad).toBe(-0.2927) // revealed
      expect(gatedNoPred.w_o_h2.grad).toBeNull() // hidden awaiting prediction
      expect(gatedNoPred.b_o.grad).toBe(-0.4534) // revealed (bias)
      expect(gatedNoPred.w_h1_x1.grad).toBeNull() // still hidden

      // 2. With prediction for w_o_h2 and attempted prediction for stage-hidden w_h1_x1
      const gatedWithPred = gatePredictions(stageWeights, {
        w_o_h2: 'decrease',
        w_h1_x1: 'increase',
      })
      expect(gatedWithPred.w_o_h1.grad).toBe(-0.2927)
      expect(gatedWithPred.w_o_h2.grad).toBe(-0.2380) // revealed after prediction
      expect(gatedWithPred.b_o.grad).toBe(-0.4534)
      expect(gatedWithPred.w_h1_x1.grad).toBeNull() // CRITICAL: remains hidden!
    })

    // Data-leak break-it verification:
    it('prevents data leak: un-gated stageWeights leaks numerical gradient, whereas gatePredictions prevents leak', () => {
      const stageWeights = {
        w_o_h2: { value: -0.6, grad: -0.2380 },
      }
      // If a component erroneously reads directly from stageWeights, the answer is leaked:
      const leakedGradient = stageWeights.w_o_h2.grad
      expect(leakedGradient).not.toBeNull()
      expect(leakedGradient).toBe(-0.2380)

      // When passing through gatePredictions without a guess, the leak is completely blocked:
      const safeWeights = gatePredictions(stageWeights, {})
      expect(safeWeights.w_o_h2.grad).toBeNull()
    })
  })

  // 7. Inspector statuses and correctness comparisons
  describe('inspectorStatus and isCorrect', () => {
    it('returns locked when stageRevealed is false', () => {
      expect(inspectorStatus('w_o_h1', false, {})).toBe('locked')
      expect(inspectorStatus('w_o_h2', false, {})).toBe('locked')
      expect(inspectorStatus('w_o_h2', false, { w_o_h2: 'decrease' })).toBe('locked')
    })

    it('returns revealed for worked example w_o_h1 when stageRevealed is true', () => {
      expect(inspectorStatus('w_o_h1', true, {})).toBe('revealed')
    })

    it('returns awaiting-prediction for predictable weight when stageRevealed is true but unpredicted', () => {
      expect(inspectorStatus('w_o_h2', true, {})).toBe('awaiting-prediction')
      expect(inspectorStatus('w_h1_x1', true, {})).toBe('awaiting-prediction')
    })

    it('returns revealed for predictable weight once predicted', () => {
      expect(inspectorStatus('w_o_h2', true, { w_o_h2: 'decrease' })).toBe('revealed')
      expect(inspectorStatus('w_h1_x1', true, { w_h1_x1: 'same' })).toBe('revealed')
    })

    it('evaluates correctness properly with isCorrect', () => {
      expect(isCorrect('decrease', 0.5)).toBe(true)
      expect(isCorrect('increase', 0.5)).toBe(false)
      expect(isCorrect('increase', -0.5)).toBe(true)
      expect(isCorrect('same', 0)).toBe(true)
      expect(isCorrect('same', -0)).toBe(true)
      expect(isCorrect('same', 1e-10)).toBe(true)
    })
  })
})
