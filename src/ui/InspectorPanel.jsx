import React from 'react'
import { PARAM_META } from '../engine/chainRule.js'

const ALL_PARAMS = [
  'w_o_h1',
  'w_o_h2',
  'b_o',
  'w_h1_x1',
  'w_h1_x2',
  'b_h1',
  'w_h2_x1',
  'w_h2_x2',
  'b_h2',
]

/**
 * Gradient Inspector Panel for Milestone 12.
 * Displays the selected parameter's value, symbolic chain-rule formula, individual numerical factors,
 * the hand-calculated product, and the autodiff engine gradient with mismatch detection.
 *
 * @param {{
 *   selectedParam: string|null,
 *   expression: Object|null,
 *   isRevealed: boolean,
 *   paramValue?: number|null,
 *   onSelectParam?: (label: string) => void
 * }} props
 */
export default function InspectorPanel({
  selectedParam,
  expression,
  isRevealed = false,
  paramValue = null,
  onSelectParam,
}) {
  const meta = selectedParam ? PARAM_META[selectedParam] : null
  const valueToDisplay = paramValue !== null && paramValue !== undefined
    ? paramValue
    : expression?.paramValue

  return (
    <section
      data-testid="inspector-panel"
      className="bg-slate-900 border border-slate-800 rounded-xl p-5 w-full flex flex-col gap-4 shadow-lg"
    >
      {/* Header and Quick Selector */}
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
        </div>

        {/* Quick parameter selection pills */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-xs text-slate-400 font-mono mr-1">Select:</span>
          {ALL_PARAMS.map((pLabel) => {
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

      {/* STATE 1: Empty state before parameter selection */}
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
            Click any connection edge on the neural network canvas above or choose a parameter from the pills above to inspect its exact chain-rule factor decomposition.
          </p>
        </div>
      )}

      {/* STATE 2: Parameter selected, but gradient has NOT been revealed yet */}
      {selectedParam && !isRevealed && (
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
                The gradient for <code className="font-mono text-amber-200">{selectedParam}</code> has not been revealed at the current step. Complete the forward pass and advance the backward-pass controls to reveal its backpropagation gradient.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* STATE 3: Parameter selected and gradient IS revealed */}
      {selectedParam && isRevealed && expression && (
        <div data-testid="inspector-details" className="flex flex-col gap-4">
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
