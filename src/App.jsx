import React, { useState } from 'react'
import { initNetwork, forward } from './engine/network.js'
import { bceLoss } from './engine/loss.js'
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
import NetworkCanvas from './ui/NetworkCanvas.jsx'
import StepControls from './ui/StepControls.jsx'
import BackwardStepControls from './ui/BackwardStepControls.jsx'

const EXAMPLE = { input: [1, 0], label: 1 }

export default function App() {
  // Persistent network instance across renders
  const [net] = useState(() => initNetwork())

  // Forward and backward stepping states
  const [stepIndex, setStepIndex] = useState(INITIAL_STEP)
  const [backwardStepIndex, setBackwardStepIndex] = useState(INITIAL_BACKWARD_STEP)

  // Full forward inference, loss computation, and autodiff backward pass
  const out = forward(net, EXAMPLE.input)
  const loss = bceLoss(out.y_hat, EXAMPLE.label)
  loss.backward()

  const fullSnapshot = buildSnapshot(net, out)
  const forwardComplete = stepIndex === MAX_STEP

  // Gated snapshot visible to the canvas
  const visibleSnapshot = {
    weights: forwardComplete
      ? revealWeightGradients(fullSnapshot.weights, backwardStepIndex)
      : revealWeightGradients(fullSnapshot.weights, INITIAL_BACKWARD_STEP),
    activations: revealActivations(fullSnapshot.activations, stepIndex),
  }

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
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-center p-6">
      <main className="max-w-4xl w-full bg-slate-900 border border-slate-800 rounded-2xl p-8 shadow-2xl flex flex-col gap-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-slate-800 pb-5 gap-4">
          <div>
            <div className="flex items-center space-x-2.5 mb-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-blue-500 animate-pulse"></span>
              <span className="text-xs uppercase tracking-wider text-blue-400 font-semibold font-mono">
                Milestone 11: Backward-Pass Step Controls
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
          fullWeights={fullSnapshot.weights}
          forwardComplete={forwardComplete}
        />

        {/* Network Canvas Section */}
        <section className="flex flex-col items-center gap-3">
          <NetworkCanvas snapshot={visibleSnapshot} />
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
      </main>
    </div>
  )
}

