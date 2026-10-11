/**
 * Geometry hit-testing utilities for canvas interactions in Gradient Lens.
 * Computes shortest distances from click points to line segments and identifies clicked edges.
 */

/**
 * Calculates the shortest Euclidean distance from a point (px, py) to a line segment (ax, ay) -> (bx, by).
 * Correctly handles endpoints and zero-length segments.
 *
 * @param {number} px - Point x coordinate
 * @param {number} py - Point y coordinate
 * @param {number} ax - Segment start x coordinate
 * @param {number} ay - Segment start y coordinate
 * @param {number} bx - Segment end x coordinate
 * @param {number} by - Segment end y coordinate
 * @returns {number} Shortest Euclidean distance
 */
export function distanceToSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax
  const dy = by - ay
  const lenSq = dx * dx + dy * dy

  // Zero-length segment (start point equals end point)
  if (lenSq === 0) {
    return Math.hypot(px - ax, py - ay)
  }

  // Projection parameter t of point P onto infinite line through AB
  const t = ((px - ax) * dx + (py - ay) * dy) / lenSq

  // Clamp t to [0, 1] to restrict to segment AB
  const clampedT = Math.max(0, Math.min(1, t))

  // Closest point on the segment
  const cx = ax + clampedT * dx
  const cy = ay + clampedT * dy

  return Math.hypot(px - cx, py - cy)
}

/**
 * Identifies the closest neural-network edge to a click point (x, y) within a given distance threshold.
 *
 * @param {number} x - Click x coordinate
 * @param {number} y - Click y coordinate
 * @param {Array<{ from: string, to: string, weightLabel: string }>} edges - Edge definitions
 * @param {Record<string, { x: number, y: number }>} positions - Neuron position map
 * @param {number} [threshold=10] - Maximum hit distance in pixels
 * @returns {string|null} Closest edge's weightLabel, or null if no edge is within threshold
 */
export function findClickedEdge(x, y, edges, positions, threshold = 10) {
  if (!edges || !positions || !Array.isArray(edges)) {
    return null
  }

  let closestLabel = null
  let minDistance = Infinity

  for (const edge of edges) {
    const fromPos = positions[edge.from]
    const toPos = positions[edge.to]

    if (!fromPos || !toPos) {
      continue
    }

    const dist = distanceToSegment(x, y, fromPos.x, fromPos.y, toPos.x, toPos.y)

    if (dist <= threshold && dist < minDistance) {
      minDistance = dist
      closestLabel = edge.weightLabel
    }
  }

  return closestLabel
}
