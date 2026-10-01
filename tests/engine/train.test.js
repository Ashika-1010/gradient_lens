import { describe, it, expect } from 'vitest'
import { initNetwork, forward, params } from '../../src/engine/network.js'
import { XOR_DATA } from '../../src/engine/xorData.js'
import { mulberry32, shuffle } from '../../src/engine/rng.js'
import { trainStep, trainEpoch, train } from '../../src/engine/train.js'

describe('Headless Training & RNG (Milestone 6)', () => {
  describe('mulberry32()', () => {
    // 1. Deterministic repeatability for a given seed
    it('generates a repeatable sequence of pseudo-random numbers for a fixed seed', () => {
      const rng1 = mulberry32(12345)
      const rng2 = mulberry32(12345)

      const seq1 = Array.from({ length: 10 }, () => rng1())
      const seq2 = Array.from({ length: 10 }, () => rng2())

      expect(seq1).toEqual(seq2)
      for (const val of seq1) {
        expect(val).toBeGreaterThanOrEqual(0)
        expect(val).toBeLessThan(1)
      }
    })

    // 2. Distinct sequences for distinct seeds
    it('generates distinct pseudo-random sequences for different seeds', () => {
      const rngA = mulberry32(1)
      const rngB = mulberry32(2)

      const seqA = Array.from({ length: 5 }, () => rngA())
      const seqB = Array.from({ length: 5 }, () => rngB())

      expect(seqA).not.toEqual(seqB)
    })
  })

  describe('shuffle()', () => {
    // 3. Immutability of original array
    it('returns a new array and does not mutate the original array', () => {
      const original = [1, 2, 3, 4, 5]
      const originalCopy = [...original]

      const rng = mulberry32(42)
      const shuffled = shuffle(original, rng)

      expect(shuffled).not.toBe(original)
      expect(original).toEqual(originalCopy)
      expect(shuffled.slice().sort()).toEqual(original.slice().sort())
    })

    // 4. Deterministic shuffling with seeded PRNG
    it('produces deterministic permutations given a seeded PRNG', () => {
      const items = ['A', 'B', 'C', 'D', 'E', 'F']

      const shuffled1 = shuffle(items, mulberry32(99))
      const shuffled2 = shuffle(items, mulberry32(99))

      expect(shuffled1).toEqual(shuffled2)
      expect(shuffled1).toHaveLength(items.length)
    })
  })

  describe('trainStep()', () => {
    // 5. Canonical sequence execution & scalar loss return
    it('executes zeroGrad -> forward -> bceLoss -> backward -> sgdStep and returns scalar loss', () => {
      const net = initNetwork()
      const sample = XOR_DATA[0] // [0, 0] -> 0
      const initialWeight = net.o.w1.data

      const loss = trainStep(net, sample, 0.1)

      expect(typeof loss).toBe('number')
      expect(Number.isFinite(loss)).toBe(true)
      expect(loss).toBeGreaterThan(0)
      // Weight should have been updated
      expect(net.o.w1.data).not.toBe(initialWeight)
    })

    // 6. Preserves gradients after step for inspection
    it('updates parameter values and leaves non-zero gradients accessible after step', () => {
      const net = initNetwork()
      const sample = { input: [1, 0], label: 1 }

      trainStep(net, sample, 0.5)

      // Gradient should still be present on parameters after update
      expect(net.o.w1.grad).not.toBe(0)
      expect(Number.isFinite(net.o.w1.grad)).toBe(true)
    })
  })

  describe('trainEpoch()', () => {
    // 7. Shuffles dataset and computes average loss across all samples
    it('iterates through all dataset examples and returns average loss across the epoch', () => {
      const net = initNetwork()
      const lr = 0.1
      const rng = mulberry32(42)

      const avgLoss = trainEpoch(net, XOR_DATA, lr, rng)

      expect(typeof avgLoss).toBe('number')
      expect(Number.isFinite(avgLoss)).toBe(true)
      expect(avgLoss).toBeGreaterThan(0)
    })
  })

  describe('train()', () => {
    // 8. Returns history array with 0-indexed epochs [0, 1, ..., epochs - 1] and avgLoss
    it('runs specified number of epochs and returns structured history array with 0-indexed epochs', () => {
      const net = initNetwork()
      const epochs = 10
      const history = train(net, XOR_DATA, {
        epochs,
        learningRate: 0.1,
        rng: mulberry32(42),
      })

      expect(history).toHaveLength(epochs)
      const epochIndices = history.map((h) => h.epoch)
      expect(epochIndices).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

      for (let i = 0; i < epochs; i++) {
        expect(history[i]).toHaveProperty('epoch', i)
        expect(typeof history[i].avgLoss).toBe('number')
        expect(Number.isFinite(history[i].avgLoss)).toBe(true)
      }
    })

    // 9. Loss decreases over training on XOR dataset
    it('reduces average loss significantly over training on XOR dataset', () => {
      const net = initNetwork()
      const history = train(net, XOR_DATA, {
        epochs: 500,
        learningRate: 1.0,
        rng: mulberry32(42),
      })

      const initialLoss = history[0].avgLoss
      const finalLoss = history[history.length - 1].avgLoss

      expect(finalLoss).toBeLessThan(initialLoss)
      expect(finalLoss).toBeLessThan(0.1)
    })

    // 10. Solves XOR problem with correct classification predictions
    it('successfully trains the 2-2-1 network to solve the XOR dataset', () => {
      const net = initNetwork()
      train(net, XOR_DATA, {
        epochs: 1000,
        learningRate: 1.0,
        rng: mulberry32(42),
      })

      // Check all 4 XOR predictions
      for (const sample of XOR_DATA) {
        const out = forward(net, sample.input)
        const pred = out.y_hat.data
        if (sample.label === 1) {
          expect(pred).toBeGreaterThan(0.5)
        } else {
          expect(pred).toBeLessThan(0.5)
        }
      }
    })

    // 11. Deterministic training reproducibility with identical seeds
    it('produces identical parameter weights and loss trajectories when given identical seeds', () => {
      const net1 = initNetwork()
      const net2 = initNetwork()

      const history1 = train(net1, XOR_DATA, {
        epochs: 50,
        learningRate: 0.5,
        rng: mulberry32(12345),
      })
      const history2 = train(net2, XOR_DATA, {
        epochs: 50,
        learningRate: 0.5,
        rng: mulberry32(12345),
      })

      expect(history1).toEqual(history2)

      const p1 = params(net1).map((p) => p.data)
      const p2 = params(net2).map((p) => p.data)
      expect(p1).toEqual(p2)
    })
  })
})
