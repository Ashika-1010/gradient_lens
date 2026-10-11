import React from 'react'
import { renderToString } from 'react-dom/server'
import { describe, it, expect } from 'vitest'
import NetworkCanvas from '../../src/ui/NetworkCanvas.jsx'
import { CANVAS_WIDTH, CANVAS_HEIGHT } from '../../src/visualization/layout.js'

describe('NetworkCanvas Component (Milestone 12)', () => {
  it('renders canvas element with defined dimensions and styles', () => {
    const mockSnapshot = {
      weights: {
        w_o_h1: { value: 0.7, grad: null },
      },
      activations: null,
    }

    const html = renderToString(
      React.createElement(NetworkCanvas, {
        snapshot: mockSnapshot,
        selectedParam: 'w_o_h1',
      })
    )

    expect(html).toContain('<canvas')
    expect(html).toContain(`width="${CANVAS_WIDTH}"`)
    expect(html).toContain(`height="${CANVAS_HEIGHT}"`)
  })

  it('renders canvas without error when snapshot is empty or selectedParam is null', () => {
    const html = renderToString(
      React.createElement(NetworkCanvas, {
        snapshot: null,
        selectedParam: null,
      })
    )

    expect(html).toContain('<canvas')
  })
})
