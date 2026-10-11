import React from 'react'
import {
  BACKWARD_STEPS,
  INITIAL_BACKWARD_STEP,
  canStepBackward,
  canStepPreviousGradient,
  canResetBackward,
  currentBackwardStepKey,
} from '../state/backwardSteps.js'

/**
 * Backward step controls for debugging and stepping through backpropagation gradients.
 * Reads only from gated visibleWeights so unpredicted gradients are not leaked.
 *
 * @param {{
 *   backwardStepIndex: number,
 *   onStep: () => void,
 *   onPrevious: () => void,
 *   onReset: () => void,
 *   visibleWeights?: Object|null,
 *   fullWeights?: Object|null,
 *   forwardComplete: boolean
 * }} props
 */
export default function BackwardStepControls({
  backwardStepIndex,
  onStep,
  onPrevious,
  onReset,
  visibleWeights,
  fullWeights,
  forwardComplete,
}) {
  const isStepBackwardEnabled = canStepBackward(backwardStepIndex, forwardComplete)
  const isStepPreviousEnabled = canStepPreviousGradient(backwardStepIndex, forwardComplete)
  const isResetBackwardEnabled = canResetBackward(backwardStepIndex)
  const activeKey = currentBackwardStepKey(backwardStepIndex)

  // Use gated visibleWeights; fallback to fullWeights if visibleWeights is not provided
  const weightsToUse = visibleWeights !== undefined ? visibleWeights : fullWeights

  let readoutText = ''

  if (!forwardComplete) {
    readoutText = 'Finish the forward pass first'
  } else if (backwardStepIndex === INITIAL_BACKWARD_STEP) {
    readoutText = 'Not started — click "Next Gradient" to begin backpropagation'
  } else if (activeKey && weightsToUse && weightsToUse[activeKey]) {
    const rawGrad = weightsToUse[activeKey].grad
    if (typeof rawGrad === 'number') {
      readoutText = `step ${backwardStepIndex + 1}/${BACKWARD_STEPS.length}: ∂L/∂${activeKey} = ${rawGrad.toFixed(4)}`
    } else {
      // Gradient computed by backward stage, but awaiting user prediction before reveal
      readoutText = `step ${backwardStepIndex + 1}/${BACKWARD_STEPS.length}: ∂L/∂${activeKey} computed — prediction required`
    }
  }

  const isHighlighted = forwardComplete && backwardStepIndex > INITIAL_BACKWARD_STEP

  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-4 bg-slate-950/80 border border-slate-800 rounded-xl p-4 w-full">
      {/* Control Buttons */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={onPrevious}
          disabled={!isStepPreviousEnabled}
          className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:bg-slate-800/50 disabled:text-slate-600 disabled:cursor-not-allowed text-slate-200 font-medium text-sm transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-slate-600 focus:ring-offset-2 focus:ring-offset-slate-900"
        >
          Previous Gradient
        </button>

        <button
          type="button"
          onClick={onStep}
          disabled={!isStepBackwardEnabled}
          className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed text-white font-medium text-sm transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-900"
        >
          Next Gradient
        </button>

        <button
          type="button"
          onClick={onReset}
          disabled={!isResetBackwardEnabled}
          className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:bg-slate-800/50 disabled:text-slate-600 disabled:cursor-not-allowed text-slate-200 font-medium text-sm transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-slate-600 focus:ring-offset-2 focus:ring-offset-slate-900"
        >
          Reset
        </button>
      </div>

      {/* Current Gradient Step Readout */}
      <div className="font-mono text-sm px-3.5 py-1.5 rounded-md bg-slate-900 border border-slate-800 text-slate-300">
        <span className={isHighlighted ? 'text-emerald-400 font-semibold' : 'text-slate-400'}>
          {readoutText}
        </span>
      </div>
    </div>
  )
}
