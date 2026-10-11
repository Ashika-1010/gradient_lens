import React from 'react'
import { renderToString } from 'react-dom/server'
import { describe, it, expect } from 'vitest'
import BackwardStepControls from '../../src/ui/BackwardStepControls.jsx'

describe('BackwardStepControls Component (Milestone 13)', () => {
  it('displays message when forward pass is not finished', () => {
    const html = renderToString(
      React.createElement(BackwardStepControls, {
        backwardStepIndex: -1,
        onStep: () => {},
        onPrevious: () => {},
        onReset: () => {},
        visibleWeights: {},
        forwardComplete: false,
      })
    )
    expect(html).toContain('Finish the forward pass first')
  })

  it('displays not started message at initial backward step', () => {
    const html = renderToString(
      React.createElement(BackwardStepControls, {
        backwardStepIndex: -1,
        onStep: () => {},
        onPrevious: () => {},
        onReset: () => {},
        visibleWeights: {},
        forwardComplete: true,
      })
    )
    expect(html).toContain('Not started')
  })

  it('shows numerical gradient for worked example w_o_h1 without prediction required', () => {
    const weights = {
      w_o_h1: { value: 0.7, grad: -0.2927 },
    }
    const html = renderToString(
      React.createElement(BackwardStepControls, {
        backwardStepIndex: 0, // w_o_h1
        onStep: () => {},
        onPrevious: () => {},
        onReset: () => {},
        visibleWeights: weights,
        forwardComplete: true,
      })
    )
    expect(html).toContain('step 1/9')
    expect(html).toContain('w_o_h1')
    expect(html).toContain('-0.2927')
    expect(html).not.toContain('prediction required')
  })

  it('indicates prediction required and hides numerical gradient when awaiting prediction', () => {
    // For w_o_h2 (step index 1), grad is null in gated visibleWeights
    const weights = {
      w_o_h1: { value: 0.7, grad: -0.2927 },
      w_o_h2: { value: -0.6, grad: null },
    }
    const html = renderToString(
      React.createElement(BackwardStepControls, {
        backwardStepIndex: 1, // w_o_h2
        onStep: () => {},
        onPrevious: () => {},
        onReset: () => {},
        visibleWeights: weights,
        forwardComplete: true,
      })
    )
    expect(html).toContain('step 2/9')
    expect(html).toContain('w_o_h2')
    expect(html).toContain('computed — prediction required')
    // Must NOT leak null as string or numerical value
    expect(html).not.toContain('null')
    expect(html).not.toContain('-0.2380')
  })

  it('reveals numerical gradient after prediction has been recorded', () => {
    const weights = {
      w_o_h1: { value: 0.7, grad: -0.2927 },
      w_o_h2: { value: -0.6, grad: -0.2380 },
    }
    const html = renderToString(
      React.createElement(BackwardStepControls, {
        backwardStepIndex: 1, // w_o_h2
        onStep: () => {},
        onPrevious: () => {},
        onReset: () => {},
        visibleWeights: weights,
        forwardComplete: true,
      })
    )
    expect(html).toContain('step 2/9')
    expect(html).toContain('w_o_h2')
    expect(html).toContain('-0.2380')
    expect(html).not.toContain('prediction required')
  })
})
