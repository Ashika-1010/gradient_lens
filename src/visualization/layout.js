/**
 * Fixed geometry and layout constants for the Gradient Lens 2->2->1 neural network.
 * Pure layout definition: contains no React, gradients, or weights.
 */

export const CANVAS_WIDTH = 600
export const CANVAS_HEIGHT = 400
export const NEURON_RADIUS = 24

export const NEURON_POSITIONS = {
  x1: { x: 80, y: 140 },
  x2: { x: 80, y: 260 },
  h1: { x: 300, y: 140 },
  h2: { x: 300, y: 260 },
  o:  { x: 520, y: 200 },
}

export const NEURONS = [
  { id: 'x1', label: 'x1', layer: 'input', x: 80, y: 140 },
  { id: 'x2', label: 'x2', layer: 'input', x: 80, y: 260 },
  { id: 'h1', label: 'h1', layer: 'hidden', x: 300, y: 140 },
  { id: 'h2', label: 'h2', layer: 'hidden', x: 300, y: 260 },
  { id: 'o',  label: 'ŷ',  layer: 'output', x: 520, y: 200 },
]

export const EDGES = [
  { from: 'x1', to: 'h1', weightLabel: 'w_h1_x1' },
  { from: 'x2', to: 'h1', weightLabel: 'w_h1_x2' },
  { from: 'x1', to: 'h2', weightLabel: 'w_h2_x1' },
  { from: 'x2', to: 'h2', weightLabel: 'w_h2_x2' },
  { from: 'h1', to: 'o',  weightLabel: 'w_o_h1' },
  { from: 'h2', to: 'o',  weightLabel: 'w_o_h2' },
]
