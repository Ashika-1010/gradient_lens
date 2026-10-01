import { initNetwork, forward, params } from '../src/engine/network.js'
import { bceLoss } from '../src/engine/loss.js'
import { XOR_DATA } from '../src/engine/xorData.js'
import { train } from '../src/engine/train.js'
import { mulberry32 } from '../src/engine/rng.js'

function evaluate(net) {
  let totalLoss = 0
  const results = []

  for (const sample of XOR_DATA) {
    const out = forward(net, sample.input)
    const loss = bceLoss(out.y_hat, sample.label)
    totalLoss += loss.data
    const pred = out.y_hat.data
    const predictedClass = pred >= 0.5 ? 1 : 0
    const isCorrect = predictedClass === sample.label

    results.push({
      input: sample.input,
      target: sample.label,
      prediction: pred,
      predictedClass,
      isCorrect,
    })
  }

  return {
    avgLoss: totalLoss / XOR_DATA.length,
    results,
    accuracy: results.filter((r) => r.isCorrect).length / results.length,
  }
}

function runDemo() {
  console.log('='.repeat(65))
  console.log('  Gradient Lens — Headless XOR Training Demo (Milestone 6)')
  console.log('='.repeat(65))

  const net = initNetwork()
  const initialEval = evaluate(net)

  console.log('\n--- INITIAL STATE (Before Training) ---')
  console.log(`Average Loss: ${initialEval.avgLoss.toFixed(5)}`)
  console.log(`Accuracy:     ${(initialEval.accuracy * 100).toFixed(1)}%`)
  console.log('\nPredictions:')
  for (const r of initialEval.results) {
    console.log(
      `  Input: [${r.input.join(', ')}] -> Target: ${r.target} | ` +
      `Pred: ${r.prediction.toFixed(4)} (Class ${r.predictedClass}) ${r.isCorrect ? '✓' : '✗'}`
    )
  }

  const epochs = 1000
  const learningRate = 1.0
  const seed = 42

  console.log(`\n--- TRAINING (${epochs} Epochs, Learning Rate: ${learningRate}, Seed: ${seed}) ---`)

  const history = train(net, XOR_DATA, {
    epochs,
    learningRate,
    rng: mulberry32(seed),
  })

  for (const { epoch, avgLoss } of history) {
    if (epoch === 0 || (epoch + 1) % 200 === 0 || epoch === epochs - 1) {
      console.log(`  Epoch ${String(epoch).padStart(4)}: Average Loss = ${avgLoss.toFixed(5)}`)
    }
  }

  const finalEval = evaluate(net)

  console.log('\n--- FINAL STATE (After Training) ---')
  console.log(`Initial Loss: ${initialEval.avgLoss.toFixed(5)} -> Final Loss: ${finalEval.avgLoss.toFixed(5)}`)
  console.log(`Accuracy:     ${(finalEval.accuracy * 100).toFixed(1)}% (${finalEval.results.filter(r => r.isCorrect).length}/${XOR_DATA.length} correct)`)

  console.log('\nFinal Predictions:')
  for (const r of finalEval.results) {
    console.log(
      `  Input: [${r.input.join(', ')}] -> Target: ${r.target} | ` +
      `Pred: ${r.prediction.toFixed(4)} (Class ${r.predictedClass}) ${r.isCorrect ? '✓' : '✗'}`
    )
  }

  console.log('\nTrained Parameters:')
  for (const p of params(net)) {
    console.log(`  ${p.label.padEnd(10)}: ${p.data.toFixed(4)}`)
  }

  console.log('\n' + '='.repeat(65))
  console.log('  XOR training demonstration complete.')
  console.log('='.repeat(65))
}

runDemo()
