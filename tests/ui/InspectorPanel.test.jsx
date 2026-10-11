import React from 'react'
import { renderToString } from 'react-dom/server'
import { describe, it, expect } from 'vitest'
import InspectorPanel from '../../src/ui/InspectorPanel.jsx'

describe('InspectorPanel Component (Milestones 12 & 13)', () => {
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
    expect(html).not.toContain('data-testid="inspector-awaiting-prediction"')
  })

  it('renders not-revealed locked message when parameter is selected but stage has not reached it', () => {
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
    expect(html).not.toContain('data-testid="inspector-awaiting-prediction"')
  })

  it('renders awaiting-prediction state with Increase, Decrease, and Stay the same buttons', () => {
    const html = renderToString(
      React.createElement(InspectorPanel, {
        selectedParam: 'w_o_h2',
        expression: null,
        isRevealed: false,
        stageRevealed: true,
        prediction: null,
        paramValue: -0.6,
      })
    )

    expect(html).toContain('data-testid="inspector-awaiting-prediction"')
    expect(html).toContain('Predict Weight Update Direction')
    expect(html).toContain('data-testid="predict-increase"')
    expect(html).toContain('Increase')
    expect(html).toContain('data-testid="predict-decrease"')
    expect(html).toContain('Decrease')
    expect(html).toContain('data-testid="predict-same"')
    expect(html).toContain('Stay the same')
    expect(html).not.toContain('data-testid="inspector-details"')
    expect(html).not.toContain('data-testid="inspector-not-revealed"')
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
    expect(html).toContain('data-testid="worked-example-badge"')
    expect(html).not.toContain('data-testid="inspector-mismatch-warning"')
  })

  it('displays prediction verdict banner when prediction was submitted and revealed', () => {
    const mockExpr = {
      label: 'w_o_h2',
      paramValue: -0.6,
      formula: '(ŷ - y) * a_h2',
      factors: [
        { label: '(ŷ - y)', name: '∂L/∂z_o', symbol: '(ŷ - y)', value: -0.4534 },
        { label: 'a_h2', name: '∂z_o/∂w_o_h2', symbol: 'a_h2', value: 0.525 },
      ],
      product: -0.238,
      engineGrad: -0.238, // negative grad -> correct direction is increase
      agrees: true,
    }

    // 1. Correct guess: increase
    const htmlCorrect = renderToString(
      React.createElement(InspectorPanel, {
        selectedParam: 'w_o_h2',
        expression: mockExpr,
        isRevealed: true,
        stageRevealed: true,
        prediction: 'increase',
        paramValue: -0.6,
      })
    )
    expect(htmlCorrect).toContain('data-testid="prediction-verdict"')
    expect(htmlCorrect).toContain('Correct prediction!')

    // 2. Incorrect guess: decrease
    const htmlIncorrect = renderToString(
      React.createElement(InspectorPanel, {
        selectedParam: 'w_o_h2',
        expression: mockExpr,
        isRevealed: true,
        stageRevealed: true,
        prediction: 'decrease',
        paramValue: -0.6,
      })
    )
    expect(htmlIncorrect).toContain('data-testid="prediction-verdict"')
    expect(htmlIncorrect).toContain('Incorrect prediction')
  })

  it('displays a visible mismatch warning when product does not agree with engine gradient', () => {
    const mismatchExpr = {
      label: 'w_h2_x1',
      paramValue: 0.3,
      formula: "(ŷ - y) * w_o_h2 * σ'(z_h2) * x1",
      factors: [
        { label: '(ŷ - y)', name: '∂L/∂z_o', symbol: '(ŷ - y)', value: -0.4534 },
        { label: 'w_o_h1', name: '∂z_o/∂a_h1', symbol: 'w_o_h1', value: 0.7 },
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
