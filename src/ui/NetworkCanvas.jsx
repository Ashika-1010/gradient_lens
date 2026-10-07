import React, { useRef, useEffect } from 'react'
import {
  CANVAS_WIDTH,
  CANVAS_HEIGHT,
  NEURON_RADIUS,
  NEURON_POSITIONS,
  NEURONS,
  EDGES,
} from '../visualization/layout.js'
import { valueToColor } from '../visualization/colorScale.js'

/**
 * Renders the 2->2->1 neural network computation graph on an HTML5 canvas.
 * Consumes only primitive data snapshots.
 *
 * @param {{ snapshot: { weights: Object, activations: Object|null } }} props
 */
export default function NetworkCanvas({ snapshot }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear previous frame
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)

    // 1. Draw Edges (Weights)
    if (snapshot && snapshot.weights) {
      for (const edge of EDGES) {
        const fromPos = NEURON_POSITIONS[edge.from]
        const toPos = NEURON_POSITIONS[edge.to]
        const weight = snapshot.weights[edge.weightLabel]

        const weightVal = weight ? weight.value : 0
        const strokeColor = valueToColor(weightVal, 2)
        const lineWidth = Math.min(Math.abs(weightVal) * 3 + 1, 8)

        ctx.beginPath()
        ctx.moveTo(fromPos.x, fromPos.y)
        ctx.lineTo(toPos.x, toPos.y)
        ctx.strokeStyle = strokeColor
        ctx.lineWidth = lineWidth
        ctx.lineCap = 'round'
        ctx.stroke()

        // Draw gradient badge at midpoint if gradient is revealed
        if (weight && weight.grad != null) {
          const midX = (fromPos.x + toPos.x) / 2
          const midY = (fromPos.y + toPos.y) / 2

          ctx.beginPath()
          ctx.arc(midX, midY, 7, 0, Math.PI * 2)
          ctx.fillStyle = valueToColor(weight.grad, 1)
          ctx.fill()
          ctx.strokeStyle = '#111827'
          ctx.lineWidth = 1
          ctx.stroke()
        }
      }
    }

    // 2. Draw Neurons (Activations & Labels)
    for (const neuron of NEURONS) {
      const pos = NEURON_POSITIONS[neuron.id]
      let activation = 0.5

      if (snapshot && snapshot.activations) {
        let val = null
        if (neuron.id === 'x1') val = snapshot.activations.x1
        else if (neuron.id === 'x2') val = snapshot.activations.x2
        else if (neuron.id === 'h1') val = snapshot.activations.a_h1
        else if (neuron.id === 'h2') val = snapshot.activations.a_h2
        else if (neuron.id === 'o') val = snapshot.activations.y_hat

        if (val !== null && val !== undefined) {
          activation = val
        }
      }

      // Neuron fill color based on activation centered at 0.5 with maxMagnitude 0.5
      const fillColor = valueToColor(activation - 0.5, 0.5)

      // Draw Neuron Circle
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, NEURON_RADIUS, 0, Math.PI * 2)
      ctx.fillStyle = fillColor
      ctx.fill()
      ctx.lineWidth = 2
      ctx.strokeStyle = '#1e293b' // slate-800 border
      ctx.stroke()

      // Draw Neuron Label
      ctx.font = 'bold 13px Inter, ui-sans-serif, system-ui, sans-serif'
      ctx.fillStyle = '#0f172a' // slate-900 for high contrast
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(neuron.label, pos.x, pos.y)
    }
  }, [snapshot])

  return (
    <div className="flex justify-center items-center bg-slate-950 p-4 rounded-xl border border-slate-800 shadow-inner">
      <canvas
        ref={canvasRef}
        width={CANVAS_WIDTH}
        height={CANVAS_HEIGHT}
        className="rounded-lg shadow-sm"
      />
    </div>
  )
}
