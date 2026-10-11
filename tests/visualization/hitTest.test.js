import { describe, it, expect } from 'vitest'
import { distanceToSegment, findClickedEdge } from '../../src/visualization/hitTest.js'
import { EDGES, NEURON_POSITIONS } from '../../src/visualization/layout.js'

describe('Hit Testing Geometry & Edge Selection (Milestone 12)', () => {
  describe('distanceToSegment', () => {
    it('returns 0 when point lies directly on the segment', () => {
      // Horizontal segment from (0, 0) to (100, 0)
      expect(distanceToSegment(50, 0, 0, 0, 100, 0)).toBe(0)
      expect(distanceToSegment(0, 0, 0, 0, 100, 0)).toBe(0)
      expect(distanceToSegment(100, 0, 0, 0, 100, 0)).toBe(0)
    })

    it('returns exact perpendicular distance when projection falls within segment', () => {
      // Horizontal segment (0, 10) to (100, 10), point at (50, 25)
      expect(distanceToSegment(50, 25, 0, 10, 100, 10)).toBe(15)
      // Vertical segment (10, 0) to (10, 100), point at (18, 40)
      expect(distanceToSegment(18, 40, 10, 0, 10, 100)).toBe(8)
    })

    it('returns distance to endpoint A when projection falls before segment start', () => {
      // Segment (10, 10) to (50, 10), point at (0, 10) -> distance to (10, 10) = 10
      expect(distanceToSegment(0, 10, 10, 10, 50, 10)).toBe(10)
      // Point at (4, 2) -> distance to (10, 10) = sqrt((4-10)^2 + (2-10)^2) = 10
      expect(distanceToSegment(4, 2, 10, 10, 50, 10)).toBe(10)
    })

    it('returns distance to endpoint B when projection falls beyond segment end', () => {
      // Segment (10, 10) to (50, 10), point at (60, 10) -> distance to (50, 10) = 10
      expect(distanceToSegment(60, 10, 10, 10, 50, 10)).toBe(10)
      // Point at (50, 16) -> distance = 6
      expect(distanceToSegment(50, 16, 10, 10, 50, 10)).toBe(6)
    })

    it('correctly handles zero-length segments without division-by-zero or NaN', () => {
      // Segment (20, 30) to (20, 30), point at (23, 34) -> distance = sqrt(3^2 + 4^2) = 5
      const dist = distanceToSegment(23, 34, 20, 30, 20, 30)
      expect(dist).toBe(5)
      expect(Number.isFinite(dist)).toBe(true)
    })

    it('calculates accurate distance for diagonal segments', () => {
      // Segment from (0, 0) to (100, 100)
      // Point at (50, 50) is on line -> 0
      expect(distanceToSegment(50, 50, 0, 0, 100, 100)).toBe(0)
      // Point at (0, 100): closest point on line is (50, 50), distance = sqrt(50^2 + 50^2) ≈ 70.7106
      expect(distanceToSegment(0, 100, 0, 0, 100, 100)).toBeCloseTo(Math.hypot(50, 50), 5)
    })
  })

  describe('findClickedEdge', () => {
    it('finds the correct edge when clicked near its midpoint', () => {
      // For edge h1 -> o (w_o_h1): h1 is (300, 140), o is (520, 200)
      // Midpoint is (410, 170)
      const hit = findClickedEdge(410, 170, EDGES, NEURON_POSITIONS, 10)
      expect(hit).toBe('w_o_h1')
    })

    it('detects hits within the specified threshold', () => {
      // Midpoint of w_o_h1 is (410, 170), test 6px offset -> within threshold 10
      const hitNear = findClickedEdge(410, 176, EDGES, NEURON_POSITIONS, 10)
      expect(hitNear).toBe('w_o_h1')
    })

    it('returns null when click is farther than threshold', () => {
      // Midpoint of w_o_h1 is (410, 170), test 25px offset
      const miss = findClickedEdge(410, 195, EDGES, NEURON_POSITIONS, 10)
      // Point (410, 195) is between w_o_h1 (y=170) and w_o_h2 (midpoint y=230)
      // Distance to w_o_h1 is 25 > 10, distance to w_o_h2 is 35 > 10
      expect(miss).toBe(null)
    })

    it('correctly resolves each of the 6 network edges when clicked along its path', () => {
      for (const edge of EDGES) {
        const from = NEURON_POSITIONS[edge.from]
        const to = NEURON_POSITIONS[edge.to]
        // Use t = 0.3 to avoid the exact diagonal intersection at t = 0.5 between w_h1_x2 and w_h2_x1
        const sampleX = from.x + 0.3 * (to.x - from.x)
        const sampleY = from.y + 0.3 * (to.y - from.y)

        const clicked = findClickedEdge(sampleX, sampleY, EDGES, NEURON_POSITIONS, 10)
        expect(clicked).toBe(edge.weightLabel)
      }
    })

    it('chooses the strictly closer edge when multiple edges are within threshold', () => {
      // Between w_h1_x1 and w_h1_x2 near neuron h1 (300, 140)
      // Point slightly closer to w_h1_x1:
      const x1Pos = NEURON_POSITIONS.x1 // (80, 140)
      const h1Pos = NEURON_POSITIONS.h1 // (300, 140)
      // Segment x1 -> h1 is perfectly horizontal at y = 140
      // Click at (190, 142) is 2px from w_h1_x1
      const hit = findClickedEdge(190, 142, EDGES, NEURON_POSITIONS, 15)
      expect(hit).toBe('w_h1_x1')
    })

    it('returns null gracefully on empty edges array or invalid inputs', () => {
      expect(findClickedEdge(100, 100, [], NEURON_POSITIONS)).toBe(null)
      expect(findClickedEdge(100, 100, null, NEURON_POSITIONS)).toBe(null)
      expect(findClickedEdge(100, 100, EDGES, null)).toBe(null)
    })
  })
})
