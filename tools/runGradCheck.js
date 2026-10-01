import { initNetwork as createNetwork, forward } from '../src/engine/network.js'
import { XOR_DATA } from '../src/engine/xorData.js'
import { compareGradients } from '../src/engine/gradCheck.js'

function runGradCheck() {
  // 1. Create a fresh network
  const net = createNetwork()

  // 2. Select XOR example: input [1, 0], label 1
  const example = XOR_DATA.find(
    (ex) => ex.input[0] === 1 && ex.input[1] === 0 && ex.label === 1
  ) || { input: [1, 0], label: 1 }

  // 3. Run numerical vs analytic comparison with eps = 1e-5
  const eps = 1e-5
  const results = compareGradients(net, example, eps)

  // 4. Print formatted table
  console.log('='.repeat(78))
  console.log('  Gradient Lens — Milestone 7: Numerical Gradient Checking')
  console.log('='.repeat(78))
  console.log(`Example: input = [${example.input.join(', ')}], label = ${example.label} | epsilon = ${eps}\n`)

  console.log(
    'Parameter'.padEnd(14) +
    'Analytic Grad'.padEnd(20) +
    'Numerical Grad'.padEnd(20) +
    'Absolute Diff'.padEnd(16) +
    'Match (<1e-4)'
  )
  console.log('-'.repeat(78))

  let maxDiff = 0

  for (const r of results) {
    const diff = r.absDiff
    if (diff > maxDiff) {
      maxDiff = diff
    }

    const match = diff < 1e-4 ? '✓ PASS' : '✗ FAIL'

    console.log(
      r.label.padEnd(14) +
      r.analytic.toFixed(8).padEnd(20) +
      r.numerical.toFixed(8).padEnd(20) +
      diff.toExponential(4).padEnd(16) +
      match
    )
  }

  console.log('-'.repeat(78))
  console.log(`Maximum Absolute Difference: ${maxDiff.toExponential(6)} (${maxDiff < 1e-4 ? 'PASSED: < 1e-4' : 'FAILED'})`)
  console.log('='.repeat(78))
}

runGradCheck()
