/**
 * Topological ordering of a computation graph rooted at `root`.
 *
 * Traverses the graph depth-first, ensuring that:
 * 1. Every reachable node is visited exactly once.
 * 2. Parents appear before their consumer nodes.
 * 3. The `root` node appears last in the list.
 *
 * @param {Value} root - The root node of the computation graph
 * @returns {Value[]} Array of Value nodes in topological order (parents -> consumers -> root)
 */
export function topologicalOrder(root) {
  const topo = []
  const visited = new Set()

  function buildTopo(node) {
    if (!visited.has(node)) {
      visited.add(node)
      for (const parent of node.parents) {
        buildTopo(parent)
      }
      topo.push(node)
    }
  }

  buildTopo(root)
  return topo
}

/**
 * Value represents a scalar node in an autodiff computation graph.
 * Stores its forward data, accumulated gradient, parent operands, operation, and backward rule.
 */
export class Value {
  /**
   * @param {number} data - Forward numerical value
   * @param {Value[]} [parents=[]] - Operands used to produce this value (preserves duplicate references)
   * @param {string} [op=''] - Symbol of operation producing this node ('+', '*', '')
   * @param {string} [label=''] - Optional human-readable display label (e.g. 'w1', 'x1')
   */
  constructor(data, parents = [], op = '', label = '') {
    this.data = Number(data)
    this.grad = 0
    this.parents = parents
    this.op = op
    this.label = label
    this._backward = () => { }
  }

  /**
   * Coerces a numeric value or existing Value into a Value instance.
   *
   * @param {number|Value} val - Number or Value instance
   * @param {string} [label=''] - Optional label if creating a new Value
   * @returns {Value}
   */
  static from(val, label = '') {
    if (val instanceof Value) {
      return val
    }
    return new Value(Number(val), [], '', label)
  }

  /**
   * Forward addition: out = this + other
   * Backward rule:
   *   ∂out/∂this = 1.0  =>  this.grad += 1.0 * out.grad
   *   ∂out/∂other = 1.0 =>  other.grad += 1.0 * out.grad
   *
   * @param {number|Value} other
   * @returns {Value}
   */
  add(other) {
    const otherVal = Value.from(other)
    const out = new Value(this.data + otherVal.data, [this, otherVal], '+')

    out._backward = () => {
      this.grad += 1.0 * out.grad
      otherVal.grad += 1.0 * out.grad
    }

    return out
  }

  /**
   * Forward multiplication: out = this * other
   * Backward rule:
   *   ∂out/∂this = other.data => this.grad += other.data * out.grad
   *   ∂out/∂other = this.data => other.grad += this.data * out.grad
   *
   * @param {number|Value} other
   * @returns {Value}
   */
  mul(other) {
    const otherVal = Value.from(other)
    const out = new Value(this.data * otherVal.data, [this, otherVal], '*')

    out._backward = () => {
      this.grad += otherVal.data * out.grad
      otherVal.grad += this.data * out.grad
    }

    return out
  }

  /**
   * Reverse-mode automatic differentiation starting from this root node.
   *
   * Steps:
   * 1. Traverse the computation graph and compute topological ordering.
   * 2. Reset gradients of all nodes in the graph to 0 (gradient hygiene).
   * 3. Seed the root node's gradient to 1.0 (∂this/∂this = 1).
   * 4. Process nodes in reverse topological order, calling each node's `_backward()`.
   */
  backward() {
    const topo = topologicalOrder(this)

    // Gradient hygiene: reset all gradients in the graph to 0
    for (const node of topo) {
      node.grad = 0
    }

    // Seed root gradient
    this.grad = 1.0

    // Reverse topological traversal: consumers are processed before their parents
    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward()
    }
  }
}
