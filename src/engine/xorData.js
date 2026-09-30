/**
 * XOR Dataset for Gradient Lens
 *
 * Classic non-linearly separable dataset:
 * - [0, 0] -> 0
 * - [0, 1] -> 1
 * - [1, 0] -> 1
 * - [1, 1] -> 0
 */
export const XOR_DATA = [
  { input: [0, 0], label: 0 },
  { input: [0, 1], label: 1 },
  { input: [1, 0], label: 1 },
  { input: [1, 1], label: 0 },
]
