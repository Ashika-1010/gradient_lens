import { describe, it, expect } from 'vitest'
import {
  valueToColor,
  POSITIVE,
  NEGATIVE,
  NEUTRAL,
} from '../../src/visualization/colorScale.js'

function parseRgb(rgbStr) {
  const match = rgbStr.match(/^rgb\((\d+),\s*(\d+),\s*(\d+)\)$/)
  if (!match) return null
  return [Number(match[1]), Number(match[2]), Number(match[3])]
}

describe('Visualization Color Scale (Milestone 9)', () => {
  // 1. Zero produces exact neutral gray
  it('returns exact neutral gray when value is zero', () => {
    const color = valueToColor(0, 1)
    expect(color).toBe(`rgb(${NEUTRAL[0]}, ${NEUTRAL[1]}, ${NEUTRAL[2]})`)
  })

  // 2. Positive values lean blue
  it('interpolates toward positive blue for positive values', () => {
    const color = valueToColor(1.0, 1.0)
    expect(color).toBe(`rgb(${POSITIVE[0]}, ${POSITIVE[1]}, ${POSITIVE[2]})`)

    const halfColor = parseRgb(valueToColor(0.5, 1.0))
    expect(halfColor).not.toBeNull()
    // Blue channel should be higher than neutral, red channel lower
    expect(halfColor[2]).toBeGreaterThan(NEUTRAL[2])
    expect(halfColor[0]).toBeLessThan(NEUTRAL[0])
  })

  // 3. Negative values lean orange
  it('interpolates toward negative orange for negative values', () => {
    const color = valueToColor(-1.0, 1.0)
    expect(color).toBe(`rgb(${NEGATIVE[0]}, ${NEGATIVE[1]}, ${NEGATIVE[2]})`)

    const halfColor = parseRgb(valueToColor(-0.5, 1.0))
    expect(halfColor).not.toBeNull()
    // Red channel should increase towards orange, blue channel decrease
    expect(halfColor[0]).toBeGreaterThan(NEUTRAL[0])
    expect(halfColor[2]).toBeLessThan(NEUTRAL[2])
  })

  // 4. Larger magnitude produces stronger saturation
  it('produces monotonically stronger color saturation for larger same-sign magnitudes', () => {
    const rgbLow = parseRgb(valueToColor(0.25, 1.0))
    const rgbMid = parseRgb(valueToColor(0.5, 1.0))
    const rgbHigh = parseRgb(valueToColor(0.75, 1.0))

    // Blue channel increases monotonically towards blue target
    expect(rgbMid[2]).toBeGreaterThan(rgbLow[2])
    expect(rgbHigh[2]).toBeGreaterThan(rgbMid[2])
  })

  // 5. Clamps values exceeding maxMagnitude to full saturation
  it('clamps values beyond maxMagnitude to full saturation', () => {
    const colorAtMax = valueToColor(1.0, 1.0)
    const colorBeyondMax = valueToColor(5.0, 1.0)
    const colorNegativeBeyond = valueToColor(-10.0, 1.0)

    expect(colorBeyondMax).toBe(colorAtMax)
    expect(colorNegativeBeyond).toBe(valueToColor(-1.0, 1.0))
  })

  // 6. Valid CSS RGB string with channels in [0, 255]
  it('always produces a valid CSS rgb string with channel values strictly between 0 and 255', () => {
    const testValues = [-100, -2, -0.5, 0, 0.5, 2, 100]

    for (const val of testValues) {
      const rgbStr = valueToColor(val, 2)
      const rgb = parseRgb(rgbStr)

      expect(rgb).not.toBeNull()
      expect(rgb[0]).toBeGreaterThanOrEqual(0)
      expect(rgb[0]).toBeLessThanOrEqual(255)
      expect(rgb[1]).toBeGreaterThanOrEqual(0)
      expect(rgb[1]).toBeLessThanOrEqual(255)
      expect(rgb[2]).toBeGreaterThanOrEqual(0)
      expect(rgb[2]).toBeLessThanOrEqual(255)
    }
  })
})
