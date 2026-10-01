/**
 * Creates a deterministic 32-bit pseudo-random number generator using the Mulberry32 algorithm.
 *
 * @param {number} seed - Integer seed value
 * @returns {() => number} Deterministic PRNG returning numbers in [0, 1)
 */
export function mulberry32(seed) {
  let s = Math.trunc(seed) >>> 0
  return function () {
    s = (s + 0x6D2B79F5) >>> 0
    let t = Math.imul(s ^ (s >>> 15), 1 | s)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Returns a new shuffled copy of an array using the Fisher-Yates shuffle algorithm.
 * The original array is not mutated.
 *
 * @template T
 * @param {T[]} array - The array to shuffle
 * @param {() => number} [rng=Math.random] - Random number generator returning [0, 1)
 * @returns {T[]} A new shuffled array
 */
export function shuffle(array, rng = Math.random) {
  const result = [...array]
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    const temp = result[i]
    result[i] = result[j]
    result[j] = temp
  }
  return result
}
