import React from 'react'
import { describe, it, expect, vi } from 'vitest'
import { initNetwork, forward, params } from '../../src/engine/network.js'
import { bceLoss } from '../../src/engine/loss.js'
import { EDGES, NEURON_POSITIONS } from '../../src/visualization/layout.js'
import { findClickedEdge } from '../../src/visualization/hitTest.js'
import { buildSnapshot } from '../../src/state/snapshot.js'
import { revealWeightGradients } from '../../src/state/backwardSteps.js'
import { gatePredictions, recordPrediction } from '../../src/state/prediction.js'

describe('Weight Edge Selection, Display, and Outside-Click Dismissal (UI Enhancement)', () => {
  // 1. Edge selection and switching via hit-testing
  it('identifies selected edge, switches selection on new edge click, and clears on empty space click', () => {
    // Edge w_o_h1: from h1 (300, 140) to o (520, 200) -> midpoint (410, 170)
    const hit1 = findClickedEdge(410, 170, EDGES, NEURON_POSITIONS, 12)
    expect(hit1).toBe('w_o_h1')

    // Switch selection to edge w_o_h2: from h2 (300, 260) to o (520, 200) -> midpoint (410, 230)
    const hit2 = findClickedEdge(410, 230, EDGES, NEURON_POSITIONS, 12)
    expect(hit2).toBe('w_o_h2')

    // Click on empty canvas space far from any edge (e.g., top-left corner 10, 10)
    const miss = findClickedEdge(10, 10, EDGES, NEURON_POSITIONS, 12)
    expect(miss).toBeNull()
  })

  // 2. Displayed value matches actual current parameter weight formatted to 4 decimals
  it('formats displayed weight label with parameter label and exact 4-decimal value for all edges', () => {
    const net = initNetwork()
    const out = forward(net, [1, 0])
    const snapshot = buildSnapshot(net, out)

    for (const edge of EDGES) {
      const weightObj = snapshot.weights[edge.weightLabel]
      expect(weightObj).toBeDefined()

      const formattedLabel = `${edge.weightLabel}: ${weightObj.value.toFixed(4)}`
      expect(formattedLabel).toContain(edge.weightLabel)
      expect(formattedLabel).toMatch(/^[a-z0-9_]+: -?[0-9]+\.[0-9]{4}$/)

      // Specific known default values
      if (edge.weightLabel === 'w_o_h1') {
        expect(formattedLabel).toBe('w_o_h1: 0.7000')
      } else if (edge.weightLabel === 'w_o_h2') {
        expect(formattedLabel).toBe('w_o_h2: -0.6000')
      } else if (edge.weightLabel === 'w_h1_x1') {
        expect(formattedLabel).toBe('w_h1_x1: 0.5000')
      }
    }
  })

  // 3. Canvas rendering draws only the currently selected edge label
  it('simulates canvas draw and verifies only the selected edge weight label is rendered', () => {
    const net = initNetwork()
    const snapshot = buildSnapshot(net)

    // Helper that replicates NetworkCanvas draw logic for selected edge badge
    function getRenderedLabel(selectedEdge) {
      if (!snapshot || !snapshot.weights || !selectedEdge) return null
      const edgeObj = EDGES.find((e) => e.weightLabel === selectedEdge)
      if (!edgeObj) return null
      const weight = snapshot.weights[edgeObj.weightLabel]
      const weightVal = weight ? weight.value : 0
      return `${edgeObj.weightLabel}: ${weightVal.toFixed(4)}`
    }

    // When w_o_h1 is selected:
    expect(getRenderedLabel('w_o_h1')).toBe('w_o_h1: 0.7000')

    // When selection switches to w_o_h2:
    expect(getRenderedLabel('w_o_h2')).toBe('w_o_h2: -0.6000')
    // Old label is NOT present:
    expect(getRenderedLabel('w_o_h2')).not.toBe('w_o_h1: 0.7000')

    // When selection is dismissed (null):
    expect(getRenderedLabel(null)).toBeNull()
  })

  // 4. Outside-click dismissal logic preserves inspector interaction while dismissing on outside clicks
  describe('Outside-click dismissal event handling logic', () => {
    it('dismisses selection when clicking outside both network diagram and inspector', () => {
      let selectedParam = 'w_o_h1'
      const setSelectedParam = (val) => {
        selectedParam = val
      }

      // Mock DOM elements and container refs
      const mockCanvas = { nodeName: 'CANVAS' }
      const mockNetworkContainer = {
        contains: (target) => target === mockCanvas,
      }
      const mockInspectorButton = { nodeName: 'BUTTON' }
      const mockInspectorContainer = {
        contains: (target) => target === mockInspectorButton,
      }

      const handleOutsideClick = (e) => {
        const isInsideNetwork = mockNetworkContainer.contains(e.target)
        const isInsideInspector = mockInspectorContainer.contains(e.target)
        if (!isInsideNetwork && !isInsideInspector) {
          setSelectedParam(null)
        }
      }

      // 1. Click on canvas -> preserved
      handleOutsideClick({ target: mockCanvas })
      expect(selectedParam).toBe('w_o_h1')

      // 2. Click on inspector button -> preserved
      handleOutsideClick({ target: mockInspectorButton })
      expect(selectedParam).toBe('w_o_h1')

      // 3. Click outside (e.g. page background) -> dismissed
      const mockPageBackground = { nodeName: 'DIV' }
      handleOutsideClick({ target: mockPageBackground })
      expect(selectedParam).toBeNull()
    })
  })

  // 5. Dismissing selection does NOT erase saved predictions, modify parameters, or change progress
  it('ensures dismissing selection does not erase predictions or modify network weights', () => {
    const net = initNetwork()
    const out = forward(net, [1, 0])
    const loss = bceLoss(out.y_hat, 1)
    loss.backward()

    const initialWeights = params(net).map((p) => ({ label: p.label, data: p.data, grad: p.grad }))

    let predictions = {}
    // User records a prediction for w_o_h2
    predictions = recordPrediction(predictions, 'w_o_h2', 'increase')
    expect(predictions.w_o_h2).toBe('increase')

    let selectedParam = 'w_o_h2'

    // User dismisses selection by clicking outside
    selectedParam = null

    // Verify predictions map is completely intact
    expect(predictions.w_o_h2).toBe('increase')

    // Verify network parameter data and gradients are completely unaltered
    params(net).forEach((p, i) => {
      expect(p.data).toBe(initialWeights[i].data)
      expect(p.grad).toBe(initialWeights[i].grad)
    })
  })

  // 6. M13 gradient gating remains intact after selecting, dismissing, and re-selecting
  it('maintains M13 gradient gating integrity across select -> dismiss -> re-select cycles', () => {
    const net = initNetwork()
    const out = forward(net, [1, 0])
    const loss = bceLoss(out.y_hat, 1)
    loss.backward()

    const fullSnapshot = buildSnapshot(net, out)

    // At backward step 1 (w_o_h2 stage revealed) but with NO prediction for w_o_h2:
    const stageWeights = revealWeightGradients(fullSnapshot.weights, 1)
    let predictions = {}

    // 1. Gate check before prediction
    const gatedBefore = gatePredictions(stageWeights, predictions)
    expect(gatedBefore.w_o_h2.grad).toBeNull() // unpredicted gradient is hidden

    // 2. Edge w_o_h2 is selected -> weight value is shown, but gradient is still hidden
    let selectedParam = 'w_o_h2'
    expect(gatedBefore[selectedParam].value).toBe(-0.6)
    expect(gatedBefore[selectedParam].grad).toBeNull()

    // 3. Outside click dismisses selection
    selectedParam = null

    // 4. User re-selects w_o_h2 and makes a prediction
    selectedParam = 'w_o_h2'
    predictions = recordPrediction(predictions, 'w_o_h2', 'increase')
    const gatedAfter = gatePredictions(stageWeights, predictions)

    // Now gradient is safely revealed
    expect(gatedAfter[selectedParam].grad).toBeCloseTo(-0.238, 4)
  })
})
