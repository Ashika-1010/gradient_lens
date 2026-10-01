import { forward } from './network.js'
import { bceLoss } from './loss.js'
import { zeroGrad, sgdStep } from './optimizer.js'
import { shuffle, mulberry32 } from './rng.js'

export { shuffle, mulberry32 }

/**
 * Executes a single training step on one dataset example.
 * Sequence: zeroGrad -> forward -> bceLoss -> backward -> sgdStep.
 *
 * @param {Object} net - Neural network containing trainable parameters
 * @param {{ input: number[], label: number }} sample - Training sample
 * @param {number} learningRate - Step size factor for SGD update
 * @returns {number} Scalar loss value for this step (loss.data)
 */
export function trainStep(net, sample, learningRate) {
  zeroGrad(net)
  const out = forward(net, sample.input)
  const loss = bceLoss(out.y_hat, sample.label)
  loss.backward()
  sgdStep(net, learningRate)
  return loss.data
}

/**
 * Runs one training epoch over the full dataset.
 * Shuffles the dataset copy using the provided RNG, steps through each example,
 * and returns the average loss across all examples.
 *
 * @param {Object} net - Neural network containing trainable parameters
 * @param {Array<{ input: number[], label: number }>} dataset - Array of training samples
 * @param {number} learningRate - Step size factor for SGD update
 * @param {(() => number)|number} [rng=Math.random] - PRNG function or numeric seed
 * @returns {number} Average loss across all samples in this epoch
 */
export function trainEpoch(net, dataset, learningRate, rng = Math.random) {
  const prng = typeof rng === 'number' ? mulberry32(rng) : rng
  const shuffled = shuffle(dataset, prng)
  let totalLoss = 0

  for (const sample of shuffled) {
    totalLoss += trainStep(net, sample, learningRate)
  }

  return dataset.length > 0 ? totalLoss / dataset.length : 0
}

/**
 * Trains the neural network for a specified number of epochs.
 *
 * @param {Object} net - Neural network containing trainable parameters
 * @param {Array<{ input: number[], label: number }>} data - Training dataset
 * @param {Object} options - Training configuration options
 * @param {number} options.epochs - Number of training epochs to run
 * @param {number} options.learningRate - Step size factor for SGD update
 * @param {() => number} [options.rng=Math.random] - Random number generator for shuffling
 * @returns {Array<{ epoch: number, avgLoss: number }>} Training history array
 */
export function train(net, data, { epochs, learningRate, rng = Math.random }) {
  const history = []

  for (let epoch = 0; epoch < epochs; epoch++) {
    const avgLoss = trainEpoch(net, data, learningRate, rng)
    history.push({ epoch, avgLoss })
  }

  return history
}
