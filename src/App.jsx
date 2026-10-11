import React, { useState, useRef, useEffect } from 'react'
import { initNetwork, forward } from './engine/network.js'
import { bceLoss } from './engine/loss.js'
import { buildChainExpression } from './engine/chainRule.js'
import { buildSnapshot } from './state/snapshot.js'
import {
  INITIAL_STEP,
  MAX_STEP,
  canStepForward,
  canStepBackward as canStepForwardBackward,
  revealActivations,
} from './state/forwardSteps.js'
import {
  INITIAL_BACKWARD_STEP,
  canStepBackward,
  canStepPreviousGradient,
  revealWeightGradients,
} from './state/backwardSteps.js'
import {
  gatePredictions,
  isGradientRevealed,
  recordPrediction,
} from './state/prediction.js'
import NetworkCanvas from './ui/NetworkCanvas.jsx'
import StepControls from './ui/StepControls.jsx'
import BackwardStepControls from './ui/BackwardStepControls.jsx'
import InspectorPanel from './ui/InspectorPanel.jsx'

const EXAMPLE = { input: [1, 0], label: 1 }

export default function App() {
  // Persistent network instance across renders
  const [net] = useState(() => initNetwork())

  // Forward and backward stepping states
  const [stepIndex, setStepIndex] = useState(INITIAL_STEP)
  const [backwardStepIndex, setBackwardStepIndex] = useState(INITIAL_BACKWARD_STEP)

  // User predictions mapped by parameter label: { [label]: 'increase'|'decrease'|'same' }
  const [predictions, setPredictions] = useState({})

  // Selected parameter state for gradient inspector and connection label
  const [selectedParam, setSelectedParam] = useState(null)

  // Container refs for detecting outside clicks
  const networkContainerRef = useRef(null)
  const inspectorContainerRef = useRef(null)

  // Handle clicks outside the network diagram and inspector to dismiss selection
  useEffect(() => {
    if (!selectedParam) return

    const handleOutsideClick = (e) => {
      const isInsideNetwork = networkContainerRef.current?.contains(e.target)
      const isInsideInspector = inspectorContainerRef.current?.contains(e.target)

      if (!isInsideNetwork && !isInsideInspector) {
        setSelectedParam(null)
      }
    }

    document.addEventListener('pointerdown', handleOutsideClick)
    document.addEventListener('mousedown', handleOutsideClick)
    document.addEventListener('click', handleOutsideClick)

    return () => {
      document.removeEventListener('pointerdown', handleOutsideClick)
      document.removeEventListener('mousedown', handleOutsideClick)
      document.removeEventListener('click', handleOutsideClick)
    }
  }, [selectedParam])

  // Full forward inference, loss computation, and autodiff backward pass
  const out = forward(net, EXAMPLE.input)
  const loss = bceLoss(out.y_hat, EXAMPLE.label)
  loss.backward()

  const fullSnapshot = buildSnapshot(net, out)
  const forwardComplete = stepIndex === MAX_STEP

  // Gate 1: Sequential backward-pass step gate
  const stageWeights = forwardComplete
    ? revealWeightGradients(fullSnapshot.weights, backwardStepIndex)
    : revealWeightGradients(fullSnapshot.weights, INITIAL_BACKWARD_STEP)

  // Gate 2: Predict -> Reveal gate (only applies to stage-revealed weights)
  const visibleWeights = gatePredictions(stageWeights, predictions)

  // Unified visible snapshot consumed by canvas and downstream views
  const visibleSnapshot = {
    weights: visibleWeights,
    activations: revealActivations(fullSnapshot.activations, stepIndex),
  }

  // Determine whether the backward stage has reached the currently selected parameter
  const stageRevealed = Boolean(
    selectedParam &&
    forwardComplete &&
    stageWeights &&
    stageWeights[selectedParam]?.grad !== null
  )

  // Determine whether the currently selected parameter is fully revealed
  const isParamRevealed = isGradientRevealed(selectedParam, stageRevealed, predictions)

  // Compute analytical chain-rule expression ONLY when parameter is revealed
  const chainExpression = isParamRevealed
    ? buildChainExpression(net, out, selectedParam, EXAMPLE.label)
    : null

  const handleForwardStep = () => {
    if (canStepForward(stepIndex)) {
      setStepIndex((prev) => prev + 1)
    }
  }

  const handleForwardPrevious = () => {
    if (canStepForwardBackward(stepIndex)) {
      setStepIndex((prev) => prev - 1)
    }
  }

  const handleForwardReset = () => {
    setStepIndex(INITIAL_STEP)
    setBackwardStepIndex(INITIAL_BACKWARD_STEP)
    setPredictions({})
  }

  const handleBackwardStep = () => {
    if (canStepBackward(backwardStepIndex, forwardComplete)) {
      setBackwardStepIndex((prev) => prev + 1)
    }
  }

  const handleBackwardPrevious = () => {
    if (canStepPreviousGradient(backwardStepIndex, forwardComplete)) {
      setBackwardStepIndex((prev) => prev - 1)
    }
  }

  const handleBackwardReset = () => {
    setBackwardStepIndex(INITIAL_BACKWARD_STEP)
    setPredictions({})
  }

  const handlePredict = (direction) => {
    if (!selectedParam) return
    setPredictions((prev) => recordPrediction(prev, selectedParam, direction))
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-center p-6">
      <main className="max-w-4xl w-full bg-slate-900 border border-slate-800 rounded-2xl p-8 shadow-2xl flex flex-col gap-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-slate-800 pb-5 gap-4">
          <div>
            <div className="flex items-center space-x-2.5 mb-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
              <span className="text-xs uppercase tracking-wider text-cyan-400 font-semibold font-mono">
                Milestone 13: Predict &rarr; Reveal
              </span>
            </div>
            <h1 className="text-3xl font-bold tracking-tight text-white">
              Gradient Lens
            </h1>
            <p className="text-slate-400 text-sm mt-0.5">
              Interactive neural-network debugger for learning backpropagation.
            </p>
          </div>

          {/* Status Metrics Pill */}
          <div className="flex items-center gap-4 bg-slate-950/80 border border-slate-800 rounded-xl px-5 py-3 font-mono text-sm shadow-sm">
            <div>
              <span className="text-xs text-slate-500 block">Example</span>
              <span className="text-slate-200 font-semibold">
                [{EXAMPLE.input.join(', ')}] &rarr; {EXAMPLE.label}
              </span>
            </div>
            <div className="w-px h-8 bg-slate-800"></div>
            <div>
              <span className="text-xs text-slate-500 block">Prediction (ŷ)</span>
              <span className="text-blue-400 font-bold">
                {out.y_hat.data.toFixed(4)}
              </span>
            </div>
            <div className="w-px h-8 bg-slate-800"></div>
            <div>
              <span className="text-xs text-slate-500 block">Loss (L)</span>
              <span className="text-orange-400 font-bold">
                {loss.data.toFixed(4)}
              </span>
            </div>
          </div>
        </div>

        {/* Forward Step Controls Section */}
        <StepControls
          stepIndex={stepIndex}
          onStep={handleForwardStep}
          onPrevious={handleForwardPrevious}
          onReset={handleForwardReset}
          fullActivations={fullSnapshot.activations}
        />

        {/* Backward Step Controls Section */}
        <BackwardStepControls
          backwardStepIndex={backwardStepIndex}
          onStep={handleBackwardStep}
          onPrevious={handleBackwardPrevious}
          onReset={handleBackwardReset}
          visibleWeights={visibleWeights}
          fullWeights={visibleWeights}
          forwardComplete={forwardComplete}
        />

        {/* Network Canvas Section */}
        <section
          ref={networkContainerRef}
          data-testid="network-container"
          className="flex flex-col items-center gap-3"
        >
          <NetworkCanvas
            snapshot={visibleSnapshot}
            selectedParam={selectedParam}
            onSelectEdge={(edgeLabel) => setSelectedParam(edgeLabel)}
          />
          <div className="flex items-center justify-center gap-6 text-xs text-slate-400 font-mono">
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-full bg-blue-600 inline-block"></span> Positive Weight / High Activation (&gt; 0.5)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-full bg-slate-300 inline-block"></span> Neutral (0.0 / 0.5)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-full bg-orange-600 inline-block"></span> Negative Weight / Low Activation (&lt; 0.5)
            </span>
          </div>
        </section>

        {/* Gradient Inspector Panel Section */}
        <div ref={inspectorContainerRef} data-testid="inspector-container">
          <InspectorPanel
            selectedParam={selectedParam}
            expression={chainExpression}
            isRevealed={isParamRevealed}
            stageRevealed={stageRevealed}
            prediction={selectedParam ? predictions[selectedParam] : null}
            onPredict={handlePredict}
            paramValue={selectedParam ? fullSnapshot.weights[selectedParam]?.value : null}
            onSelectParam={(param) => setSelectedParam(param)}
          />
        </div>
      </main>
    </div>
  )
}
