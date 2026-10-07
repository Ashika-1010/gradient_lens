import { describe, it, expect } from 'vitest'
import {
  FORWARD_STEPS,
  INITIAL_STEP,
  MAX_STEP,
  canStepForward,
  canStepBackward,
  canStepPrevious,
  canReset,
  currentStepKey,
  revealActivations,
} from '../../src/state/forwardSteps.js'

describe('Forward-Pass Step Controls (Milestone 10 & 11 UX)', () => {
  const fullActivations = {
    x1: 1.0,
    x2: 0.0,
    z_h1: 0.6,
    a_h1: 0.645656,
    z_h2: 0.1,
    a_h2: 0.524979,
    z_o: 0.186972,
    y_hat: 0.546604,
  }

  // 1. Step index bounds & ordering
  it('defines INITIAL_STEP before all real steps and MAX_STEP corresponding to y_hat', () => {
    expect(INITIAL_STEP).toBe(-1)
    expect(MAX_STEP).toBe(FORWARD_STEPS.length - 1)
    expect(FORWARD_STEPS[MAX_STEP]).toBe('y_hat')
    expect(FORWARD_STEPS).toEqual([
      'z_h1',
      'a_h1',
      'z_h2',
      'a_h2',
      'z_o',
      'y_hat',
    ])
  })

  // 2. Stepping predicates at initial boundary
  it('allows stepping forward but disallows reset and previous step at INITIAL_STEP', () => {
    expect(canStepForward(INITIAL_STEP)).toBe(true)
    expect(canReset(INITIAL_STEP)).toBe(false)
    expect(canStepBackward(INITIAL_STEP)).toBe(false)
    expect(canStepPrevious(INITIAL_STEP)).toBe(false)
  })

  // 3. Stepping predicates at final boundary
  it('disallows stepping forward but allows reset and previous step at MAX_STEP', () => {
    expect(canStepForward(MAX_STEP)).toBe(false)
    expect(canReset(MAX_STEP)).toBe(true)
    expect(canStepBackward(MAX_STEP)).toBe(true)
    expect(canStepPrevious(MAX_STEP)).toBe(true)
  })

  // 4. Stepping predicates in intermediate steps
  it('allows both stepping forward, reset, and previous step in intermediate steps', () => {
    for (let step = 0; step < MAX_STEP; step++) {
      expect(canStepForward(step)).toBe(true)
      expect(canReset(step)).toBe(true)
      expect(canStepBackward(step)).toBe(true)
      expect(canStepPrevious(step)).toBe(true)
    }
  })

  // 5. Stepping previous predicates at specific boundary steps
  it('evaluates canStepBackward correctly at step 0 and MAX_STEP (step 5)', () => {
    expect(canStepBackward(0)).toBe(true)
    expect(canStepBackward(5)).toBe(true)
    expect(canStepBackward(-1)).toBe(false)
  })

  // 6. Decrementing step index behavior
  it('correctly handles decrementing from step 0 to INITIAL_STEP and step 5 to 4', () => {
    let stepIndex = 0
    if (canStepBackward(stepIndex)) {
      stepIndex -= 1
    }
    expect(stepIndex).toBe(-1)
    expect(canStepBackward(stepIndex)).toBe(false)

    stepIndex = 5
    if (canStepBackward(stepIndex)) {
      stepIndex -= 1
    }
    expect(stepIndex).toBe(4)
    expect(canStepBackward(stepIndex)).toBe(true)
  })

  // 7. currentStepKey lookup
  it('returns null at INITIAL_STEP and matches FORWARD_STEPS for every valid index', () => {
    expect(currentStepKey(INITIAL_STEP)).toBeNull()
    expect(currentStepKey(-99)).toBeNull()
    expect(currentStepKey(99)).toBeNull()

    FORWARD_STEPS.forEach((key, idx) => {
      expect(currentStepKey(idx)).toBe(key)
    })
  })

  // 8. revealActivations at INITIAL_STEP
  it('reveals x1 and x2 while gating all computed activations to null at INITIAL_STEP', () => {
    const revealed = revealActivations(fullActivations, INITIAL_STEP)

    expect(revealed.x1).toBe(1.0)
    expect(revealed.x2).toBe(0.0)
    expect(revealed.z_h1).toBeNull()
    expect(revealed.a_h1).toBeNull()
    expect(revealed.z_h2).toBeNull()
    expect(revealed.a_h2).toBeNull()
    expect(revealed.z_o).toBeNull()
    expect(revealed.y_hat).toBeNull()
  })

  // 9. revealActivations at step 0 (z_h1)
  it('reveals through z_h1 at step 0 while keeping subsequent activations null', () => {
    const revealed = revealActivations(fullActivations, 0)

    expect(revealed.x1).toBe(1.0)
    expect(revealed.x2).toBe(0.0)
    expect(revealed.z_h1).toBe(0.6)
    expect(revealed.a_h1).toBeNull()
    expect(revealed.z_h2).toBeNull()
    expect(revealed.a_h2).toBeNull()
    expect(revealed.z_o).toBeNull()
    expect(revealed.y_hat).toBeNull()
  })

  // 10. revealActivations at intermediate step 2 (z_h1, a_h1, z_h2)
  it('reveals through z_h2 at step 2 while keeping a_h2 and later activations null', () => {
    const revealed = revealActivations(fullActivations, 2)

    expect(revealed.x1).toBe(1.0)
    expect(revealed.x2).toBe(0.0)
    expect(revealed.z_h1).toBe(0.6)
    expect(revealed.a_h1).toBe(0.645656)
    expect(revealed.z_h2).toBe(0.1)
    expect(revealed.a_h2).toBeNull()
    expect(revealed.z_o).toBeNull()
    expect(revealed.y_hat).toBeNull()
  })

  // 11. revealActivations at MAX_STEP
  it('reveals all activations at MAX_STEP matching the complete forward activations object', () => {
    const revealed = revealActivations(fullActivations, MAX_STEP)

    expect(revealed).toEqual(fullActivations)
  })

  // 12. revealActivations when stepping back from step 5 to 4
  it('correctly re-hides step 5 activation (y_hat) when stepping backward from step 5 to 4', () => {
    const step5 = revealActivations(fullActivations, 5)
    expect(step5.y_hat).toBe(0.546604)
    expect(step5.z_o).toBe(0.186972)

    const step4 = revealActivations(fullActivations, 4)
    expect(step4.y_hat).toBeNull()
    expect(step4.z_o).toBe(0.186972)
    expect(step4.a_h2).toBe(0.524979)
  })

  // 13. Immutability of input activations
  it('guarantees that revealActivations does not mutate the fullActivations object', () => {
    const originalCopy = { ...fullActivations }

    revealActivations(fullActivations, 1)

    expect(fullActivations).toEqual(originalCopy)
  })
})
