import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import { describe, it, expect } from 'vitest'
import { initNetwork, forward, params } from '../../src/engine/network.js'
import { bceLoss } from '../../src/engine/loss.js'
import { zeroGrad } from '../../src/engine/optimizer.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const fixturePath = path.resolve(__dirname, '../fixtures/pytorch_reference.json')
const reference = JSON.parse(fs.readFileSync(fixturePath, 'utf-8'))

describe('PyTorch Autograd Cross-Check (Milestone 8)', () => {
  // 1. Fixture sanity tests
  describe('Fixture Sanity', () => {
    it('verifies that the PyTorch reference fixture has expected loss and output weight gradient', () => {
      expect(reference.loss).toBeCloseTo(0.6040, 3)
      expect(reference.gradients.w_o_h1).toBeCloseTo(-0.2927, 4)
      expect(reference.input).toEqual([1.0, 0.0])
      expect(reference.label).toBe(1.0)
    })
  })

  // 2. Forward pass comparisons against PyTorch reference
  describe('Forward Pass Cross-Check', () => {
    it('matches PyTorch intermediate pre-activations (z_h1, z_h2, z_o) to 6 decimal places', () => {
      const net = initNetwork()
      const out = forward(net, reference.input)

      expect(out.z_h1.data).toBeCloseTo(reference.forward.z_h1, 6)
      expect(out.z_h2.data).toBeCloseTo(reference.forward.z_h2, 6)
      expect(out.z_o.data).toBeCloseTo(reference.forward.z_o, 6)
    })

    it('matches PyTorch activations (a_h1, a_h2) and output prediction y_hat to 6 decimal places', () => {
      const net = initNetwork()
      const out = forward(net, reference.input)

      expect(out.a_h1.data).toBeCloseTo(reference.forward.a_h1, 6)
      expect(out.a_h2.data).toBeCloseTo(reference.forward.a_h2, 6)
      expect(out.y_hat.data).toBeCloseTo(reference.forward.y_hat, 6)
    })
  })

  // 3. Loss calculation comparison against PyTorch reference
  describe('Loss Calculation Cross-Check', () => {
    it('matches PyTorch binary cross-entropy loss to 6 decimal places', () => {
      const net = initNetwork()
      const out = forward(net, reference.input)
      const loss = bceLoss(out.y_hat, reference.label)

      expect(loss.data).toBeCloseTo(reference.loss, 6)
    })
  })

  // 4. Reverse-mode autodiff gradient cross-check against PyTorch autograd
  describe('Autodiff Gradient Cross-Check', () => {
    it('matches PyTorch autograd gradients for all hidden layer 1 parameters to 6 decimal places', () => {
      const net = initNetwork()
      zeroGrad(net)
      const out = forward(net, reference.input)
      const loss = bceLoss(out.y_hat, reference.label)
      loss.backward()

      expect(net.h1.w1.grad).toBeCloseTo(reference.gradients.w_h1_x1, 6)
      expect(net.h1.w2.grad).toBeCloseTo(reference.gradients.w_h1_x2, 6)
      expect(net.h1.b.grad).toBeCloseTo(reference.gradients.b_h1, 6)
    })

    it('matches PyTorch autograd gradients for all hidden layer 2 parameters to 6 decimal places', () => {
      const net = initNetwork()
      zeroGrad(net)
      const out = forward(net, reference.input)
      const loss = bceLoss(out.y_hat, reference.label)
      loss.backward()

      expect(net.h2.w1.grad).toBeCloseTo(reference.gradients.w_h2_x1, 6)
      expect(net.h2.w2.grad).toBeCloseTo(reference.gradients.w_h2_x2, 6)
      expect(net.h2.b.grad).toBeCloseTo(reference.gradients.b_h2, 6)
    })

    it('matches PyTorch autograd gradients for all output layer parameters and all 9 parameters to 6 decimal places', () => {
      const net = initNetwork()
      zeroGrad(net)
      const out = forward(net, reference.input)
      const loss = bceLoss(out.y_hat, reference.label)
      loss.backward()

      expect(net.o.w1.grad).toBeCloseTo(reference.gradients.w_o_h1, 6)
      expect(net.o.w2.grad).toBeCloseTo(reference.gradients.w_o_h2, 6)
      expect(net.o.b.grad).toBeCloseTo(reference.gradients.b_o, 6)

      // Comprehensive iteration over all 9 trainable parameters
      const allParams = params(net)
      expect(allParams).toHaveLength(9)
      for (const p of allParams) {
        expect(p.grad).toBeCloseTo(reference.gradients[p.label], 6)
      }
    })
  })
})
