import React from 'react'
import { PARAM_META } from '../engine/chainRule.js'
import {
  PREDICTABLE_LABELS,
  WORKED_EXAMPLE_LABEL,
  correctDirection,
  isCorrect,
  isPredictionRequired,
  inspectorStatus,
} from '../state/prediction.js'

// The 6 selectable edge weights (biases are out of scope for prediction/inspection)
const SELECTABLE_WEIGHTS = [
  'w_o_h1',
  'w_o_h2',
  'w_h1_x1',
  'w_h1_x2',
  'w_h2_x1',
  'w_h2_x2',
]

/**
 * Gradient Inspector Panel for Milestone 13: Predict -> Reveal.
 *
 * Supports three distinct states for a selected parameter:
 * 1. 'locked': Backward pass has not reached this parameter yet.
 * 2. 'awaiting-prediction': Backward pass reached this parameter; prediction required before revealing gradient.
 * 3. 'revealed': Gradient and chain factors visible (either worked example or prediction submitted).
 *
 * @param {{
 *   selectedParam: string|null,
 *   expression?: Object|null,
 *   isRevealed?: boolean,
 *   stageRevealed?: boolean,
 *   status?: string|null,
 *   prediction?: string|null,
 *   onPredict?: (direction: string) => void,
 *   paramValue?: number|null,
 *   onSelectParam?: (label: string) => void
 * }} props
 */
export default function InspectorPanel({
  selectedParam,
  expression = null,
  isRevealed = false,
  stageRevealed = false,
  status = null,
  prediction = null,
  onPredict = null,
  paramValue = null,
  onSelectParam = null,
}) {
  const meta = selectedParam ? PARAM_META[selectedParam] : null
  const valueToDisplay =
    paramValue !== null && paramValue !== undefined
      ? paramValue
      : expression?.paramValue

  // Determine current inspector status
  let currentStatus = status
  if (!currentStatus) {
    if (!selectedParam) {
      currentStatus = null
    } else if (isRevealed) {
      currentStatus = 'revealed'
    } else {
      currentStatus = inspectorStatus(
        selectedParam,
        stageRevealed,
        prediction ? { [selectedParam]: prediction } : {}
      )
    }
  }

  const isWorkedExample = selectedParam === WORKED_EXAMPLE_LABEL
  const predictionNeeded = selectedParam && isPredictionRequired(selectedParam)

  return (
    <section
      data-testid="inspector-panel"
      className="bg-slate-900 border border-slate-800 rounded-xl p-5 w-full flex flex-col gap-4 shadow-lg"
    >
      {/* Header and Edge Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-800 gap-3">
        <div className="flex items-center gap-2.5">
          <div className="w-2.5 h-2.5 rounded-full bg-cyan-400"></div>
          <h2 className="text-base font-semibold text-white tracking-wide">
            Gradient Inspector
          </h2>
          {selectedParam && (
            <span className="font-mono text-xs px-2 py-0.5 rounded bg-cyan-950/80 text-cyan-300 border border-cyan-800/80">
              {selectedParam}
            </span>
          )}
          {isWorkedExample && currentStatus === 'revealed' && (
            <span
              data-testid="worked-example-badge"
              className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-800/80"
            >
              Worked Example
            </span>
          )}
        </div>

        {/* Edge parameter selection pills */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-xs text-slate-400 font-mono mr-1">Select:</span>
          {SELECTABLE_WEIGHTS.map((pLabel) => {
            const isCurrent = selectedParam === pLabel
            return (
              <button
                key={pLabel}
                type="button"
                onClick={() => onSelectParam && onSelectParam(pLabel)}
                className={`text-xs font-mono px-2 py-0.5 rounded transition-all ${
                  isCurrent
                    ? 'bg-cyan-500 text-slate-950 font-bold shadow-sm'
                    : 'bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700/60'
                }`}
              >
                {pLabel}
              </button>
            )
          })}
        </div>
      </div>

      {/* STATE 0: Empty state before parameter selection */}
      {!selectedParam && (
        <div
          data-testid="inspector-empty"
          className="flex flex-col items-center justify-center py-8 px-4 text-center border border-dashed border-slate-800 rounded-lg bg-slate-950/40"
        >
          <div className="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center mb-3 text-slate-400 text-lg">
            🔍
          </div>
          <h3 className="text-sm font-semibold text-slate-200">
            No parameter selected
          </h3>
          <p className="text-xs text-slate-400 max-w-md mt-1">
            Click any connection edge on the neural network canvas above or choose a weight from the pills above to inspect its exact chain-rule factor decomposition.
          </p>
        </div>
      )}

      {/* STATE 1: Locked state (stage gate has not reached this parameter) */}
      {selectedParam && currentStatus === 'locked' && (
        <div
          data-testid="inspector-not-revealed"
          className="flex flex-col gap-3 p-4 rounded-lg bg-amber-950/20 border border-amber-800/40"
        >
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs text-amber-300 font-semibold uppercase tracking-wider">
              {meta?.displayName || selectedParam}
            </span>
            {valueToDisplay !== null && valueToDisplay !== undefined && (
              <span className="font-mono text-xs text-slate-300">
                Current Value: <strong className="text-white">{valueToDisplay.toFixed(4)}</strong>
              </span>
            )}
          </div>

          <div className="flex items-start gap-3 mt-1">
            <span className="text-amber-400 text-lg leading-none mt-0.5">⏳</span>
            <div>
              <p className="text-sm font-medium text-amber-200">
                Gradient not revealed yet
              </p>
              <p className="text-xs text-amber-400/80 mt-0.5">
                The gradient for <code className="font-mono text-amber-200">{selectedParam}</code> has not been revealed at the current step. Complete the forward pass and advance the backward-pass controls to reach this parameter.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* STATE 2: Awaiting Prediction state (stage reached parameter, prediction required) */}
      {selectedParam && currentStatus === 'awaiting-prediction' && (
        <div
          data-testid="inspector-awaiting-prediction"
          className="flex flex-col gap-4 p-5 rounded-lg bg-blue-950/30 border border-blue-800/50"
        >
          <div className="flex items-center justify-between border-b border-blue-900/50 pb-2">
            <div>
              <span className="font-mono text-xs text-blue-300 font-semibold uppercase tracking-wider">
                {meta?.displayName || selectedParam}
              </span>
              <span className="text-xs text-slate-400 ml-2">
                ({meta?.description || 'Trainable weight'})
              </span>
            </div>
            {valueToDisplay !== null && valueToDisplay !== undefined && (
              <span className="font-mono text-xs text-slate-300">
                Current Value: <strong className="text-white font-mono">{valueToDisplay.toFixed(4)}</strong>
              </span>
            )}
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <span className="text-blue-400 text-base">🎯</span>
              <h3 className="text-sm font-semibold text-white">
                Predict Weight Update Direction
              </h3>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Under Gradient Descent, weights update according to <code className="font-mono text-cyan-300 bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800">w_new = w - η · (∂L/∂w)</code>.
              Based on the network state, will this weight <strong className="text-white">increase</strong>, <strong className="text-white">decrease</strong>, or <strong className="text-white">stay the same</strong>?
            </p>
          </div>

          {/* Interactive Prediction Buttons */}
          <div className="flex flex-wrap items-center gap-3 pt-1">
            <button
              type="button"
              data-testid="predict-increase"
              onClick={() => onPredict && onPredict('increase')}
              className="flex-1 min-w-[120px] px-4 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs sm:text-sm transition-all shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:ring-offset-2 focus:ring-offset-slate-900 flex items-center justify-center gap-1.5 cursor-pointer"
            >
              <span>Increase</span>
              <span className="text-emerald-200">(&uarr;)</span>
            </button>

            <button
              type="button"
              data-testid="predict-decrease"
              onClick={() => onPredict && onPredict('decrease')}
              className="flex-1 min-w-[120px] px-4 py-2.5 rounded-lg bg-orange-600 hover:bg-orange-500 text-white font-medium text-xs sm:text-sm transition-all shadow-sm focus:outline-none focus:ring-2 focus:ring-orange-400 focus:ring-offset-2 focus:ring-offset-slate-900 flex items-center justify-center gap-1.5 cursor-pointer"
            >
              <span>Decrease</span>
              <span className="text-orange-200">(&darr;)</span>
            </button>

            <button
              type="button"
              data-testid="predict-same"
              onClick={() => onPredict && onPredict('same')}
              className="flex-1 min-w-[120px] px-4 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-100 font-medium text-xs sm:text-sm transition-all shadow-sm focus:outline-none focus:ring-2 focus:ring-slate-500 focus:ring-offset-2 focus:ring-offset-slate-900 border border-slate-700 flex items-center justify-center gap-1.5 cursor-pointer"
            >
              <span>Stay the same</span>
              <span className="text-slate-400">(=)</span>
            </button>
          </div>
        </div>
      )}

      {/* STATE 3: Revealed state (worked example or prediction submitted) */}
      {selectedParam && currentStatus === 'revealed' && expression && (
        <div data-testid="inspector-details" className="flex flex-col gap-4">
          {/* Prediction Verdict Banner (only shown if a prediction was required and submitted) */}
          {predictionNeeded && prediction && (
            <div
              data-testid="prediction-verdict"
              className={`p-3.5 rounded-lg border flex flex-col gap-1.5 font-sans ${
                isCorrect(prediction, expression.engineGrad)
                  ? 'bg-emerald-950/40 border-emerald-700/60 text-emerald-200'
                  : 'bg-rose-950/40 border-rose-700/60 text-rose-200'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-semibold text-sm flex items-center gap-1.5">
                  {isCorrect(prediction, expression.engineGrad) ? (
                    <>
                      <span>✓</span>
                      <span>Correct prediction!</span>
                    </>
                  ) : (
                    <>
                      <span>✗</span>
                      <span>Incorrect prediction</span>
                    </>
                  )}
                </span>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-950/80 border border-slate-800 text-slate-300">
                  Your guess: <strong className="text-white capitalize">{prediction === 'same' ? 'Stay the same' : prediction}</strong>
                </span>
              </div>
              <p className="text-xs opacity-90 leading-relaxed">
                Gradient <code className="font-mono font-semibold px-1 py-0.5 rounded bg-slate-950/60">∂L/∂{selectedParam} = {expression.engineGrad.toFixed(4)}</code>.
                With gradient descent update rule <code className="font-mono px-1 py-0.5 rounded bg-slate-950/60">w_new = w - η·g</code>, the weight will{' '}
                <strong className="underline">
                  {correctDirection(expression.engineGrad) === 'same'
                    ? 'stay the same'
                    : correctDirection(expression.engineGrad)}
                </strong>.
              </p>
            </div>
          )}

          {/* Parameter Summary Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-950/60 border border-slate-800/80 rounded-lg p-3">
            <div>
              <span className="text-xs text-slate-400 block font-mono">Parameter</span>
              <span className="text-sm font-semibold text-white font-mono">
                {selectedParam}
              </span>
              <span className="text-xs text-slate-400 ml-2">
                ({meta?.description || 'Trainable parameter'})
              </span>
            </div>

            <div>
              <span className="text-xs text-slate-400 block font-mono">Current Value</span>
              <span className="text-sm font-bold text-cyan-300 font-mono">
                {valueToDisplay !== null && valueToDisplay !== undefined
                  ? valueToDisplay.toFixed(4)
                  : 'N/A'}
              </span>
            </div>

            <div>
              <span className="text-xs text-slate-400 block font-mono">Engine Gradient</span>
              <span
                data-testid="inspector-engine-grad"
                className="text-sm font-bold text-blue-400 font-mono"
              >
                {expression.engineGrad !== null && expression.engineGrad !== undefined
                  ? expression.engineGrad.toFixed(4)
                  : 'N/A'}
              </span>
            </div>
          </div>

          {/* Chain-Rule Symbolic Formula */}
          <div className="bg-slate-950 border border-slate-800 rounded-lg p-3">
            <span className="text-xs text-slate-400 block font-mono mb-1">
              Chain-Rule Expression
            </span>
            <div
              data-testid="inspector-formula"
              className="font-mono text-sm sm:text-base text-amber-300 font-semibold tracking-wide"
            >
              ∂L/∂{selectedParam} = {expression.formula}
            </div>
          </div>

          {/* Individual Numerical Factors Breakdown */}
          <div>
            <span className="text-xs font-mono text-slate-400 block mb-2">
              Individual Factors Breakdown ({expression.factors.length} {expression.factors.length === 1 ? 'factor' : 'factors'})
            </span>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-2.5">
              {expression.factors.map((factor, idx) => (
                <div
                  key={`${factor.label}-${idx}`}
                  className="bg-slate-950/80 border border-slate-800 rounded-lg p-2.5 flex flex-col justify-between shadow-sm"
                >
                  <div className="flex items-center justify-between text-xs text-slate-400 font-mono mb-1">
                    <span className="text-slate-300 font-semibold">{factor.symbol || factor.label}</span>
                    <span className="text-[10px] text-slate-400">{factor.name}</span>
                  </div>
                  <div className="text-base font-mono font-bold text-slate-100">
                    {factor.value.toFixed(4)}
                  </div>
                  <div className="text-[11px] text-slate-400 truncate mt-1">
                    {factor.description || factor.name}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Calculation Comparison and Verification */}
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 p-3.5 rounded-lg bg-slate-950 border border-slate-800">
            <div className="flex items-center gap-4 font-mono text-sm">
              <div>
                <span className="text-xs text-slate-400 block">Factor Product</span>
                <span data-testid="inspector-product" className="font-bold text-white">
                  {expression.product.toFixed(4)}
                </span>
              </div>
              <div className="text-slate-600 text-lg">=</div>
              <div>
                <span className="text-xs text-slate-400 block">Autodiff Engine</span>
                <span className="font-bold text-blue-400">
                  {expression.engineGrad !== null ? expression.engineGrad.toFixed(4) : 'N/A'}
                </span>
              </div>
            </div>

            {/* Mismatch Warning vs Verification Success */}
            {!expression.agrees && expression.engineGrad !== null && (
              <div
                data-testid="inspector-mismatch-warning"
                className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-rose-950/70 border border-rose-700 text-rose-200 text-xs font-semibold"
              >
                <span>⚠</span>
                <span>
                  Mismatch warning: Product ({expression.product.toFixed(6)}) differs from engine ({expression.engineGrad.toFixed(6)})
                </span>
              </div>
            )}

            {expression.agrees && (
              <div
                data-testid="inspector-match-success"
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-emerald-950/60 border border-emerald-700/60 text-emerald-300 text-xs font-medium font-mono"
              >
                <span>✓</span>
                <span>Verified: Product matches autodiff engine (Δ &lt; 1e-6)</span>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  )
}
