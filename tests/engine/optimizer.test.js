import { describe, it, expect } from 'vitest'
import { topologicalOrder } from '../../src/engine/value.js'
import { initNetwork, forward, params } from '../../src/engine/network.js'
import { bceLoss } from '../../src/engine/loss.js'
import { zeroGrad, sgdStep } from '../../src/engine/optimizer.js'

describe('Optimizer: zeroGrad and sgdStep (Milestone 5)', () => {
  describe('zeroGrad()', () => {
    // 1. Resets every parameter gradient to exactly 0 after manual non-zero assignment
    it('resets every parameter gradient to exactly 0 when manually assigned non-zero values', () => {
      const net = initNetwork()
      for (const p of params(net)) {
        p.grad = 7.5
      }

      zeroGrad(net)

      for (const p of params(net)) {
        expect(p.grad).toBe(0)
      }
    })

    // 2. Resets all parameter gradients to 0 after backward propagation
    it('resets all parameter gradients to 0 after backward propagation', () => {
      const net = initNetwork()
      const out = forward(net, [1, 1])
      const loss = bceLoss(out.y_hat, 0)
      loss.backward()

      // Confirm gradients are non-zero before zeroGrad
      for (const p of params(net)) {
        expect(p.grad).not.toBe(0)
      }

      zeroGrad(net)

      for (const p of params(net)) {
        expect(p.grad).toBe(0)
      }
    })

    // 3. zeroGrad does not modify parameter data
    it('does not modify parameter data values', () => {
      const net = initNetwork()
      const beforeData = params(net).map((p) => p.data)

      for (const p of params(net)) {
        p.grad = 3.14
      }

      zeroGrad(net)

      const afterData = params(net).map((p) => p.data)
      expect(afterData).toEqual(beforeData)
    })
  })

  describe('sgdStep()', () => {
    // 4. Hand-calculated update for [1, 0], y = 1, lr = 0.5
    it('updates output weight matching the hand-calculated check: w1.grad ≈ -0.2927, w1.data ≈ 0.84635', () => {
      const net = initNetwork()
      const out = forward(net, [1, 0])
      const loss = bceLoss(out.y_hat, 1)
      loss.backward()

      // Initial weight is 0.7
      expect(net.o.w1.data).toBe(0.7)
      expect(net.o.w1.grad).toBeCloseTo(-0.2927, 4)

      sgdStep(net, 0.5)

      // Updated weight: 0.7 - (0.5 * -0.292738) ≈ 0.84637 (close to 0.84635)
      expect(net.o.w1.data).toBeCloseTo(0.84635, 4)
    })

    // 5. Updates all 9 network parameters according to p.data -= lr * p.grad
    it('updates all 9 network parameters according to p.data -= learningRate * p.grad', () => {
      const net = initNetwork()
      const out = forward(net, [1, 1])
      const loss = bceLoss(out.y_hat, 0)
      loss.backward()

      const lr = 0.1
      const initialDataAndGrad = params(net).map((p) => ({
        data: p.data,
        grad: p.grad,
      }))

      sgdStep(net, lr)

      const currentParams = params(net)
      for (let i = 0; i < currentParams.length; i++) {
        const expected = initialDataAndGrad[i].data - lr * initialDataAndGrad[i].grad
        expect(currentParams[i].data).toBeCloseTo(expected, 10)
      }
    })

    // 6. Verify sgdStep does NOT reset or modify gradients
    it('preserves parameter gradients and does not reset or modify them during sgdStep', () => {
      const net = initNetwork()
      const out = forward(net, [1, 0])
      const loss = bceLoss(out.y_hat, 1)
      loss.backward()

      const gradsBefore = params(net).map((p) => p.grad)
      const gradW1Before = net.o.w1.grad

      sgdStep(net, 0.5)

      expect(net.o.w1.grad).toBe(gradW1Before)
      const gradsAfter = params(net).map((p) => p.grad)
      expect(gradsAfter).toEqual(gradsBefore)
    })

    // 7. Zero learning rate produces zero change in parameter data
    it('leaves all parameter data unchanged when learning rate is 0', () => {
      const net = initNetwork()
      const out = forward(net, [1, 0])
      const loss = bceLoss(out.y_hat, 1)
      loss.backward()

      const dataBefore = params(net).map((p) => p.data)

      sgdStep(net, 0)

      const dataAfter = params(net).map((p) => p.data)
      expect(dataAfter).toEqual(dataBefore)
    })

    // 8. Learning-rate proportionality: 0.2 update is exactly twice 0.1 update
    it('scales parameter updates strictly proportional to the learning rate', () => {
      const netA = initNetwork()
      const outA = forward(netA, [1, 0])
      const lossA = bceLoss(outA.y_hat, 1)
      lossA.backward()

      const netB = initNetwork()
      const outB = forward(netB, [1, 0])
      const lossB = bceLoss(outB.y_hat, 1)
      lossB.backward()

      const initData = params(netA).map((p) => p.data)

      sgdStep(netA, 0.1)
      sgdStep(netB, 0.2)

      const dataA = params(netA).map((p) => p.data)
      const dataB = params(netB).map((p) => p.data)

      for (let i = 0; i < initData.length; i++) {
        const deltaA = dataA[i] - initData[i]
        const deltaB = dataB[i] - initData[i]
        expect(deltaB).toBeCloseTo(deltaA * 2, 10)
      }
    })
  })

  describe('Gradient Accumulation & ZeroGrad Discipline', () => {
    // Helper executing reverse-mode AD graph traversal preserving parameter gradients (as in PyTorch/micrograd)
    function backwardAccumulate(loss, net) {
      const topo = topologicalOrder(loss)
      const paramSet = new Set(params(net))
      for (const node of topo) {
        if (!paramSet.has(node)) {
          node.grad = 0
        }
      }
      loss.grad = 1.0
      for (let i = topo.length - 1; i >= 0; i--) {
        topo[i]._backward()
      }
    }

    // 9. Gradient accumulation bug without zeroGrad doubles gradients on second backward()
    it('demonstrates gradient doubling on consecutive backward passes without zeroGrad', () => {
      const net = initNetwork()
      const out1 = forward(net, [1, 0])
      const loss1 = bceLoss(out1.y_hat, 1)

      backwardAccumulate(loss1, net)
      const correctGrad = net.o.w1.grad

      // Demonstrate += accumulation behavior at network scale without zeroing gradients
      backwardAccumulate(loss1, net)

      expect(net.o.w1.grad).toBeCloseTo(correctGrad * 2, 6)
    })

    // 10. Correct zeroGrad discipline resets gradients between backward passes
    it('demonstrates correct gradient computation when zeroGrad is called between backward passes', () => {
      const net = initNetwork()
      const out1 = forward(net, [1, 0])
      const loss1 = bceLoss(out1.y_hat, 1)

      backwardAccumulate(loss1, net)
      const correctGrad = net.o.w1.grad

      // Calling zeroGrad resets accumulated gradients back to 0 before re-evaluating
      zeroGrad(net)
      backwardAccumulate(loss1, net)

      expect(net.o.w1.grad).toBeCloseTo(correctGrad, 6)
    })
  })

  describe('Single Training Step', () => {
    // 11. One full training step reduces the loss
    it('reduces loss after a single small gradient-descent step (lr = 0.1)', () => {
      const net = initNetwork()

      // Initial loss before update
      const outBefore = forward(net, [1, 0])
      const lossBefore = bceLoss(outBefore.y_hat, 1)

      // Execute canonical training step sequence
      zeroGrad(net)
      const out = forward(net, [1, 0])
      const loss = bceLoss(out.y_hat, 1)
      loss.backward()
      sgdStep(net, 0.1)

      // Compute loss after update
      const outAfter = forward(net, [1, 0])
      const lossAfter = bceLoss(outAfter.y_hat, 1)

      expect(lossAfter.data).toBeLessThan(lossBefore.data)
    })
  })
})
