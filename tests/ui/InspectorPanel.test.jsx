import React from 'react'
import { renderToString } from 'react-dom/server'
import { describe, it, expect } from 'vitest'
import InspectorPanel from '../../src/ui/InspectorPanel.jsx'

describe('InspectorPanel Component (Milestone 12)', () => {
  it('renders empty state when no parameter is selected', () => {
    const html = renderToString(
      React.createElement(InspectorPanel, {
        selectedParam: null,
        expression: null,
        isRevealed: false,
      })
    )

    expect(html).toContain('data-testid="inspector-empty"')
    expect(html).toContain('No parameter selected')
    expect(html).not.toContain('data-testid="inspector-details"')
    expect(html).not.toContain('data-testid="inspector-not-revealed"')
  })

  it('renders not-revealed message when parameter is selected but gradient is not revealed', () => {
    const html = renderToString(
      React.createElement(InspectorPanel, {
        selectedParam: 'w_o_h1',
        expression: null,
        isRevealed: false,
        paramValue: 0.7,
      })
    )

    expect(html).toContain('data-testid="inspector-not-revealed"')
    expect(html).toContain('Gradient not revealed yet')
    expect(html).toContain('0.7000')
    expect(html).not.toContain('data-testid="inspector-empty"')
    expect(html).not.toContain('data-testid="inspector-details"')
  })

  it('renders complete breakdown when gradient is revealed and matching', () => {
    const mockExpr = {
      label: 'w_o_h1',
      paramValue: 0.7,
      formula: '(ŷ - y) * a_h1',
      factors: [
        { label: '(ŷ - y)', name: '∂L/∂z_o', symbol: '(ŷ - y)', description: 'Output error', value: -0.4534 },
        { label: 'a_h1', name: '∂z_o/∂w_o_h1', symbol: 'a_h1', description: 'Hidden activation h1', value: 0.6457 },
      ],
      product: -0.2927,
      engineGrad: -0.2927,
      agrees: true,
    }

    const html = renderToString(
      React.createElement(InspectorPanel, {
        selectedParam: 'w_o_h1',
        expression: mockExpr,
        isRevealed: true,
        paramValue: 0.7,
      })
    )

    expect(html).toContain('data-testid="inspector-details"')
    expect(html).toContain('data-testid="inspector-formula"')
    expect(html).toContain('(ŷ - y) * a_h1')
    expect(html).toContain('-0.4534')
    expect(html).toContain('0.6457')
    expect(html).toContain('-0.2927')
    expect(html).toContain('data-testid="inspector-match-success"')
    expect(html).not.toContain('data-testid="inspector-mismatch-warning"')
  })

  it('displays a visible mismatch warning when product does not agree with engine gradient', () => {
    const mismatchExpr = {
      label: 'w_h2_x1',
      paramValue: 0.3,
      formula: "(ŷ - y) * w_o_h2 * σ'(z_h2) * x1",
      factors: [
        { label: '(ŷ - y)', name: '∂L/∂z_o', symbol: '(ŷ - y)', value: -0.4534 },
        { label: 'w_o_h1', name: '∂z_o/∂a_h1', symbol: 'w_o_h1', value: 0.7 }, // Wrong weight
        { label: "σ'(z_h2)", name: '∂a_h2/∂z_h2', symbol: "σ'(z_h2)", value: 0.2494 },
        { label: 'x1', name: '∂z_h2/∂w_h2_x1', symbol: 'x1', value: 1.0 },
      ],
      product: -0.0791,
      engineGrad: 0.0678,
      agrees: false,
    }

    const html = renderToString(
      React.createElement(InspectorPanel, {
        selectedParam: 'w_h2_x1',
        expression: mismatchExpr,
        isRevealed: true,
        paramValue: 0.3,
      })
    )

    expect(html).toContain('data-testid="inspector-mismatch-warning"')
    expect(html).toContain('Mismatch warning')
    expect(html).not.toContain('data-testid="inspector-match-success"')
  })
})
