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
import { findClickedEdge } from '../visualization/hitTest.js'

/**
 * Renders the 2->2->1 neural network computation graph on an HTML5 canvas.
 * Consumes only primitive data snapshots and handles edge selection.
 *
 * @param {{
 *   snapshot: { weights: Object, activations: Object|null },
 *   selectedEdge?: string|null,
 *   selectedParam?: string|null,
 *   onSelectEdge?: (label: string|null) => void
 * }} props
 */
export default function NetworkCanvas({
  snapshot,
  selectedEdge = null,
  selectedParam = null,
  onSelectEdge = null,
}) {
  const canvasRef = useRef(null)
  const currentSelected = selectedParam || selectedEdge

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
        const isSelected = currentSelected === edge.weightLabel

        const weightVal = weight ? weight.value : 0
        const strokeColor = valueToColor(weightVal, 2)
        const lineWidth = Math.min(Math.abs(weightVal) * 3 + 1, 8)

        // Draw highlight glow if edge is selected
        if (isSelected) {
          ctx.save()
          ctx.beginPath()
          ctx.moveTo(fromPos.x, fromPos.y)
          ctx.lineTo(toPos.x, toPos.y)
          ctx.strokeStyle = '#38bdf8' // Sky-blue highlight glow
          ctx.lineWidth = lineWidth + 8
          ctx.lineCap = 'round'
          ctx.shadowColor = '#38bdf8'
          ctx.shadowBlur = 12
          ctx.stroke()
          ctx.restore()
        }

        // Draw main edge line
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

          // Highlight ring around badge if selected
          if (isSelected) {
            ctx.beginPath()
            ctx.arc(midX, midY, 11, 0, Math.PI * 2)
            ctx.strokeStyle = '#38bdf8'
            ctx.lineWidth = 2
            ctx.stroke()
          }

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
  }, [snapshot, currentSelected])

  const handleClick = (e) => {
    const canvas = canvasRef.current
    if (!canvas || !onSelectEdge) return

    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const clickX = (e.clientX - rect.left) * scaleX
    const clickY = (e.clientY - rect.top) * scaleY

    const clickedEdge = findClickedEdge(clickX, clickY, EDGES, NEURON_POSITIONS, 12)
    onSelectEdge(clickedEdge)
  }

  const handleMouseMove = (e) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const mouseX = (e.clientX - rect.left) * scaleX
    const mouseY = (e.clientY - rect.top) * scaleY

    const hovered = findClickedEdge(mouseX, mouseY, EDGES, NEURON_POSITIONS, 12)
    canvas.style.cursor = hovered ? 'pointer' : 'default'
  }

  return (
    <div className="flex justify-center items-center bg-slate-950 p-4 rounded-xl border border-slate-800 shadow-inner">
      <canvas
        ref={canvasRef}
        width={CANVAS_WIDTH}
        height={CANVAS_HEIGHT}
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        className="rounded-lg shadow-sm"
      />
    </div>
  )
}
