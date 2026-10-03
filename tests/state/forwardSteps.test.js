import { describe, it, expect } from 'vitest'
import {
  FORWARD_STEPS,
  INITIAL_STEP,
  MAX_STEP,
  canStepForward,
  canReset,
  currentStepKey,
  revealActivations,
} from '../../src/state/forwardSteps.js'

describe('Forward-Pass Step Controls (Milestone 10)', () => {
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
  it('allows stepping forward but disallows reset at INITIAL_STEP', () => {
    expect(canStepForward(INITIAL_STEP)).toBe(true)
    expect(canReset(INITIAL_STEP)).toBe(false)
  })

  // 3. Stepping predicates at final boundary
  it('disallows stepping forward but allows reset at MAX_STEP', () => {
    expect(canStepForward(MAX_STEP)).toBe(false)
    expect(canReset(MAX_STEP)).toBe(true)
  })

  // 4. Stepping predicates in intermediate steps
  it('allows both stepping forward and reset in intermediate steps', () => {
    for (let step = 0; step < MAX_STEP; step++) {
      expect(canStepForward(step)).toBe(true)
      expect(canReset(step)).toBe(true)
    }
  })

  // 5. currentStepKey lookup
  it('returns null at INITIAL_STEP and matches FORWARD_STEPS for every valid index', () => {
    expect(currentStepKey(INITIAL_STEP)).toBeNull()
    expect(currentStepKey(-99)).toBeNull()
    expect(currentStepKey(99)).toBeNull()

    FORWARD_STEPS.forEach((key, idx) => {
      expect(currentStepKey(idx)).toBe(key)
    })
  })

  // 6. revealActivations at INITIAL_STEP
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

  // 7. revealActivations at step 0 (z_h1)
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

  // 8. revealActivations at intermediate step 2 (z_h1, a_h1, z_h2)
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

  // 9. revealActivations at MAX_STEP
  it('reveals all activations at MAX_STEP matching the complete forward activations object', () => {
    const revealed = revealActivations(fullActivations, MAX_STEP)

    expect(revealed).toEqual(fullActivations)
  })

  // 10. Immutability of input activations
  it('guarantees that revealActivations does not mutate the fullActivations object', () => {
    const originalCopy = { ...fullActivations }

    revealActivations(fullActivations, 1)

    expect(fullActivations).toEqual(originalCopy)
  })
})
