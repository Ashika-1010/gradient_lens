import React from 'react'
import {
  canStepForward,
  canReset,
  currentStepKey,
  INITIAL_STEP,
  FORWARD_STEPS,
} from '../state/forwardSteps.js'

/**
 * Step controls for debugging and stepping through the forward propagation pass.
 *
 * @param {{
 *   stepIndex: number,
 *   onStep: () => void,
 *   onReset: () => void,
 *   fullActivations: Object|null
 * }} props
 */
export default function StepControls({
  stepIndex,
  onStep,
  onReset,
  fullActivations,
}) {
  const isStepForwardEnabled = canStepForward(stepIndex)
  const isResetEnabled = canReset(stepIndex)
  const activeKey = currentStepKey(stepIndex)

  let readoutText = 'Not started — click "Next step" to begin the forward pass'

  if (stepIndex > INITIAL_STEP && activeKey && fullActivations) {
    const rawValue = fullActivations[activeKey]
    const formattedValue =
      typeof rawValue === 'number' ? rawValue.toFixed(4) : String(rawValue)
    readoutText = `step ${stepIndex + 1}/${FORWARD_STEPS.length}: ${activeKey} = ${formattedValue}`
  }

  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-4 bg-slate-950/80 border border-slate-800 rounded-xl p-4 w-full">
      {/* Control Buttons */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={onStep}
          disabled={!isStepForwardEnabled}
          className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed text-white font-medium text-sm transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-900"
        >
          Next step
        </button>

        <button
          type="button"
          onClick={onReset}
          disabled={!isResetEnabled}
          className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:bg-slate-800/50 disabled:text-slate-600 disabled:cursor-not-allowed text-slate-200 font-medium text-sm transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-slate-600 focus:ring-offset-2 focus:ring-offset-slate-900"
        >
          Reset
        </button>
      </div>

      {/* Current Step Readout */}
      <div className="font-mono text-sm px-3.5 py-1.5 rounded-md bg-slate-900 border border-slate-800 text-slate-300">
        <span className={stepIndex > INITIAL_STEP ? 'text-emerald-400 font-semibold' : 'text-slate-400'}>
          {readoutText}
        </span>
      </div>
    </div>
  )
}
