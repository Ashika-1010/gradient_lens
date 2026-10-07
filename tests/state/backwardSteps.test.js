import { describe, it, expect } from 'vitest'
import {
  BACKWARD_STEPS,
  INITIAL_BACKWARD_STEP,
  MAX_BACKWARD_STEP,
  canStepBackward,
  canStepPreviousGradient,
  canStepPrevious,
  canStepBackwardPass,
  canResetBackward,
  currentBackwardStepKey,
  revealWeightGradients,
} from '../../src/state/backwardSteps.js'

describe('Backward-Pass Step Controls (Milestone 11)', () => {
  const fullWeights = {
    w_o_h1:  { value: 0.7,   grad: -0.2927 },
    w_o_h2:  { value: -0.6,  grad: -0.2381 },
    b_o:     { value: -0.1,  grad: -0.4534 },
    w_h1_x1: { value: 0.5,   grad: -0.0165 },
    w_h1_x2: { value: -0.3,  grad: 0.0000 },
    b_h1:    { value: 0.1,   grad: -0.0165 },
    w_h2_x1: { value: 0.8,   grad: 0.0125 },
    w_h2_x2: { value: 0.2,   grad: 0.0000 },
    b_h2:    { value: -0.7,  grad: 0.0125 },
  }

  // 1. MAX_BACKWARD_STEP definition
  it('defines MAX_BACKWARD_STEP as BACKWARD_STEPS.length - 1', () => {
    expect(INITIAL_BACKWARD_STEP).toBe(-1)
    expect(MAX_BACKWARD_STEP).toBe(BACKWARD_STEPS.length - 1)
    expect(MAX_BACKWARD_STEP).toBe(8)
  })

  // 2. BACKWARD_STEPS count and uniqueness
  it('contains exactly 9 unique parameter labels in BACKWARD_STEPS', () => {
    expect(BACKWARD_STEPS).toHaveLength(9)
    const uniqueSteps = new Set(BACKWARD_STEPS)
    expect(uniqueSteps.size).toBe(9)
    expect(BACKWARD_STEPS).toEqual([
      'w_o_h1',
      'w_o_h2',
      'b_o',
      'w_h1_x1',
      'w_h1_x2',
      'b_h1',
      'w_h2_x1',
      'w_h2_x2',
      'b_h2',
    ])
  })

  // 3. BACKWARD_STEPS matches full weight fixture keys
  it('matches all keys of the fullWeights fixture', () => {
    const fixtureKeys = Object.keys(fullWeights)
    expect(BACKWARD_STEPS.slice().sort()).toEqual(fixtureKeys.sort())
  })

  // 4. canStepBackward is false when forward is incomplete
  it('returns false for canStepBackward when forward is incomplete regardless of step index', () => {
    expect(canStepBackward(INITIAL_BACKWARD_STEP, false)).toBe(false)
    expect(canStepBackward(0, false)).toBe(false)
    expect(canStepBackward(4, false)).toBe(false)
  })

  // 5. canStepBackward is true in the middle once forward is complete
  it('returns true for canStepBackward before reaching MAX_BACKWARD_STEP once forward is complete', () => {
    expect(canStepBackward(INITIAL_BACKWARD_STEP, true)).toBe(true)
    for (let step = 0; step < MAX_BACKWARD_STEP; step++) {
      expect(canStepBackward(step, true)).toBe(true)
    }
  })

  // 6. canStepBackward is false at MAX_BACKWARD_STEP
  it('returns false for canStepBackward at MAX_BACKWARD_STEP even when forward is complete', () => {
    expect(canStepBackward(MAX_BACKWARD_STEP, true)).toBe(false)
  })

  // 7. canStepPreviousGradient predicate at boundaries and aliases
  it('evaluates canStepPreviousGradient correctly at INITIAL_BACKWARD_STEP (-1), step 0, and MAX_BACKWARD_STEP (8)', () => {
    expect(canStepPreviousGradient(INITIAL_BACKWARD_STEP)).toBe(false)
    expect(canStepPreviousGradient(INITIAL_BACKWARD_STEP, true)).toBe(false)
    expect(canStepPrevious(INITIAL_BACKWARD_STEP)).toBe(false)
    expect(canStepBackwardPass(INITIAL_BACKWARD_STEP)).toBe(false)

    expect(canStepPreviousGradient(0)).toBe(true)
    expect(canStepPreviousGradient(0, true)).toBe(true)
    expect(canStepPrevious(0)).toBe(true)
    expect(canStepBackwardPass(0)).toBe(true)

    expect(canStepPreviousGradient(8)).toBe(true)
    expect(canStepPreviousGradient(8, true)).toBe(true)
    expect(canStepPrevious(8)).toBe(true)
    expect(canStepBackwardPass(8)).toBe(true)
  })

  // 8. canStepPreviousGradient forward completion gating
  it('returns false for canStepPreviousGradient when forward is incomplete regardless of step index', () => {
    expect(canStepPreviousGradient(INITIAL_BACKWARD_STEP, false)).toBe(false)
    expect(canStepPreviousGradient(0, false)).toBe(false)
    expect(canStepPreviousGradient(4, false)).toBe(false)
    expect(canStepPreviousGradient(8, false)).toBe(false)
  })

  // 9. Decrementing backward step index behavior
  it('correctly handles decrementing from step 0 to INITIAL_BACKWARD_STEP and step 8 to 7', () => {
    let backwardStepIndex = 0
    if (canStepPreviousGradient(backwardStepIndex, true)) {
      backwardStepIndex -= 1
    }
    expect(backwardStepIndex).toBe(-1)
    expect(canStepPreviousGradient(backwardStepIndex, true)).toBe(false)

    backwardStepIndex = 8
    if (canStepPreviousGradient(backwardStepIndex, true)) {
      backwardStepIndex -= 1
    }
    expect(backwardStepIndex).toBe(7)
    expect(canStepPreviousGradient(backwardStepIndex, true)).toBe(true)
  })

  // 10. canResetBackward predicate
  it('returns false for canResetBackward at INITIAL_BACKWARD_STEP and true thereafter', () => {
    expect(canResetBackward(INITIAL_BACKWARD_STEP)).toBe(false)
    for (let step = 0; step <= MAX_BACKWARD_STEP; step++) {
      expect(canResetBackward(step)).toBe(true)
    }
  })

  // 11. currentBackwardStepKey lookup
  it('returns null at INITIAL_BACKWARD_STEP and matches BACKWARD_STEPS for valid indices', () => {
    expect(currentBackwardStepKey(INITIAL_BACKWARD_STEP)).toBeNull()
    expect(currentBackwardStepKey(-99)).toBeNull()
    expect(currentBackwardStepKey(99)).toBeNull()

    BACKWARD_STEPS.forEach((key, idx) => {
      expect(currentBackwardStepKey(idx)).toBe(key)
    })
  })

  // 12. revealWeightGradients always preserves parameter values
  it('always preserves all parameter values across all backward steps', () => {
    for (let step = INITIAL_BACKWARD_STEP; step <= MAX_BACKWARD_STEP; step++) {
      const revealed = revealWeightGradients(fullWeights, step)
      for (const label of BACKWARD_STEPS) {
        expect(revealed[label].value).toBe(fullWeights[label].value)
      }
    }
  })

  // 13. revealWeightGradients hides every gradient initially
  it('hides every gradient at INITIAL_BACKWARD_STEP (-1)', () => {
    const revealed = revealWeightGradients(fullWeights, INITIAL_BACKWARD_STEP)

    for (const label of BACKWARD_STEPS) {
      expect(revealed[label].value).toBe(fullWeights[label].value)
      expect(revealed[label].grad).toBeNull()
    }
  })

  // 14. step 0 reveals only w_o_h1 gradient
  it('reveals only w_o_h1 gradient at step 0', () => {
    const revealed = revealWeightGradients(fullWeights, 0)

    expect(revealed.w_o_h1.grad).toBe(-0.2927)
    expect(revealed.w_o_h2.grad).toBeNull()
    expect(revealed.b_o.grad).toBeNull()
    expect(revealed.w_h1_x1.grad).toBeNull()
    expect(revealed.w_h1_x2.grad).toBeNull()
    expect(revealed.b_h1.grad).toBeNull()
    expect(revealed.w_h2_x1.grad).toBeNull()
    expect(revealed.w_h2_x2.grad).toBeNull()
    expect(revealed.b_h2.grad).toBeNull()
  })

  // 15. step 2 reveals w_o_h1, w_o_h2, and b_o
  it('reveals w_o_h1, w_o_h2, and b_o gradients at step 2', () => {
    const revealed = revealWeightGradients(fullWeights, 2)

    expect(revealed.w_o_h1.grad).toBe(-0.2927)
    expect(revealed.w_o_h2.grad).toBe(-0.2381)
    expect(revealed.b_o.grad).toBe(-0.4534)
    expect(revealed.w_h1_x1.grad).toBeNull()
    expect(revealed.w_h1_x2.grad).toBeNull()
    expect(revealed.b_h1.grad).toBeNull()
    expect(revealed.w_h2_x1.grad).toBeNull()
    expect(revealed.w_h2_x2.grad).toBeNull()
    expect(revealed.b_h2.grad).toBeNull()
  })

  // 16. MAX_BACKWARD_STEP reveals all gradients
  it('reveals all 9 parameter gradients at MAX_BACKWARD_STEP', () => {
    const revealed = revealWeightGradients(fullWeights, MAX_BACKWARD_STEP)

    for (const label of BACKWARD_STEPS) {
      expect(revealed[label].value).toBe(fullWeights[label].value)
      expect(revealed[label].grad).toBe(fullWeights[label].grad)
    }
    expect(revealed).toEqual(fullWeights)
  })

  // 17. revealWeightGradients when stepping backward from step 8 to 7
  it('correctly re-hides step 8 gradient (b_h2) when stepping backward from step 8 to 7', () => {
    const step8 = revealWeightGradients(fullWeights, 8)
    expect(step8.b_h2.grad).toBe(0.0125)
    expect(step8.w_h2_x2.grad).toBe(0.0000)

    const step7 = revealWeightGradients(fullWeights, 7)
    expect(step7.b_h2.grad).toBeNull()
    expect(step7.w_h2_x2.grad).toBe(0.0000)
    expect(step7.w_h2_x1.grad).toBe(0.0125)
  })

  // 18. Immutability: revealWeightGradients does not mutate the original object
  it('does not mutate the original fullWeights object', () => {
    const copy = JSON.parse(JSON.stringify(fullWeights))
    revealWeightGradients(fullWeights, 3)
    expect(fullWeights).toEqual(copy)
  })
})
