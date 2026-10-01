import { describe, it, expect } from 'vitest'
import {
  CANVAS_WIDTH,
  CANVAS_HEIGHT,
  NEURON_RADIUS,
  NEURON_POSITIONS,
  NEURONS,
  EDGES,
} from '../../src/visualization/layout.js'

describe('Visualization Layout & Geometry (Milestone 9)', () => {
  // 1. Fixed canvas dimensions and neuron radius
  it('defines the exact fixed canvas geometry and radius constants', () => {
    expect(CANVAS_WIDTH).toBe(600)
    expect(CANVAS_HEIGHT).toBe(400)
    expect(NEURON_RADIUS).toBe(24)
  })

  // 2. Exactly 6 edges
  it('defines exactly 6 directed edges connecting layers', () => {
    expect(EDGES).toHaveLength(6)
  })

  // 3. Every edge references valid neuron position IDs
  it('ensures every edge references valid source and target neuron IDs in NEURON_POSITIONS', () => {
    const validIds = Object.keys(NEURON_POSITIONS)

    for (const edge of EDGES) {
      expect(validIds).toContain(edge.from)
      expect(validIds).toContain(edge.to)
      expect(edge.weightLabel).toBeDefined()
      expect(typeof edge.weightLabel).toBe('string')
    }
  })

  // 4. All weight labels are unique
  it('ensures all edge weightLabels are distinct and unique', () => {
    const labels = EDGES.map((e) => e.weightLabel)
    const uniqueLabels = new Set(labels)
    expect(uniqueLabels.size).toBe(6)
  })

  // 5. Layer progression in x-coordinate: inputs < hidden < output
  it('positions neurons with strictly increasing x-coordinates from input to hidden to output layers', () => {
    const x1Pos = NEURON_POSITIONS.x1.x
    const x2Pos = NEURON_POSITIONS.x2.x
    const h1Pos = NEURON_POSITIONS.h1.x
    const h2Pos = NEURON_POSITIONS.h2.x
    const oPos  = NEURON_POSITIONS.o.x

    expect(x1Pos).toBe(x2Pos)
    expect(h1Pos).toBe(h2Pos)
    expect(x1Pos).toBeLessThan(h1Pos)
    expect(h1Pos).toBeLessThan(oPos)
  })
})
