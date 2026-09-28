import { describe, it, expect } from 'vitest'
import { Value, topologicalOrder } from '../../src/engine/value.js'

describe('Scalar Autodiff Value Engine (Milestone 1)', () => {
  // 1. Value construction and initial fields
  it('constructs a Value node with expected initial fields', () => {
    const v = new Value(4.5, [], '', 'v_init')
    expect(v.data).toBe(4.5)
    expect(v.grad).toBe(0)
    expect(v.parents).toEqual([])
    expect(v.op).toBe('')
    expect(v.label).toBe('v_init')
    expect(typeof v._backward).toBe('function')
  })

  // 2. Coercion of plain numbers using Value.from
  it('coerces numbers and preserves existing Value instances via Value.from', () => {
    const existing = new Value(3.14, [], '', 'pi')
    const fromExisting = Value.from(existing)
    expect(fromExisting).toBe(existing)

    const fromNumber = Value.from(42, 'answer')
    expect(fromNumber).toBeInstanceOf(Value)
    expect(fromNumber.data).toBe(42)
    expect(fromNumber.grad).toBe(0)
    expect(fromNumber.label).toBe('answer')
  })

  // 3. Addition forward pass
  it('computes forward addition correctly and sets parents & op', () => {
    const a = new Value(3, [], '', 'a')
    const b = new Value(5, [], '', 'b')
    const c = a.add(b)

    expect(c.data).toBe(8)
    expect(c.op).toBe('+')
    expect(c.parents).toEqual([a, b])
  })

  // 4. Addition backward pass
  it('computes gradients for addition correctly', () => {
    const a = new Value(3, [], '', 'a')
    const b = new Value(5, [], '', 'b')
    const c = a.add(b)

    c.backward()

    expect(c.grad).toBe(1.0)
    expect(a.grad).toBe(1.0)
    expect(b.grad).toBe(1.0)
  })

  // 5. Multiplication forward pass
  it('computes forward multiplication correctly and sets parents & op', () => {
    const a = new Value(3, [], '', 'a')
    const b = new Value(4, [], '', 'b')
    const c = a.mul(b)

    expect(c.data).toBe(12)
    expect(c.op).toBe('*')
    expect(c.parents).toEqual([a, b])
  })

  // 6. Multiplication backward pass
  it('computes gradients for multiplication correctly via product rule', () => {
    // c = a * b => dc/da = b, dc/db = a
    const a = new Value(3, [], '', 'a')
    const b = new Value(4, [], '', 'b')
    const c = a.mul(b)

    c.backward()

    expect(c.grad).toBe(1.0)
    expect(a.grad).toBe(4) // b.data
    expect(b.grad).toBe(3) // a.data
  })

  // 7. Composition / chain rule
  it('computes gradients through chained addition and multiplication', () => {
    // L = (a + b) * (c + d)
    // a=2, b=3 => e=5
    // c=4, d=1 => f=5
    // L = e * f = 25
    // dL/de = f = 5, dL/df = e = 5
    // dL/da = dL/de * de/da = 5 * 1 = 5
    // dL/db = 5 * 1 = 5
    // dL/dc = dL/df * df/dc = 5 * 1 = 5
    // dL/dd = 5 * 1 = 5
    const a = new Value(2, [], '', 'a')
    const b = new Value(3, [], '', 'b')
    const c = new Value(4, [], '', 'c')
    const d = new Value(1, [], '', 'd')

    const e = a.add(b)
    const f = c.add(d)
    const L = e.mul(f)

    expect(L.data).toBe(25)

    L.backward()

    expect(L.grad).toBe(1.0)
    expect(e.grad).toBe(5)
    expect(f.grad).toBe(5)
    expect(a.grad).toBe(5)
    expect(b.grad).toBe(5)
    expect(c.grad).toBe(5)
    expect(d.grad).toBe(5)
  })

  // 8. a * a gradient accumulation
  it('correctly accumulates gradients when multiplying a node by itself (a * a)', () => {
    // c = a * a => dc/da = 2a
    const a = new Value(3, [], '', 'a')
    const c = a.mul(a)

    // Parents array must retain duplicate references [a, a]
    expect(c.parents).toEqual([a, a])
    expect(c.parents.length).toBe(2)
    expect(c.data).toBe(9)

    c.backward()

    expect(c.grad).toBe(1.0)
    expect(a.grad).toBe(6) // 2 * 3 = 6
  })

  // 9. a * b + a gradient accumulation
  it('accumulates gradients correctly for a * b + a', () => {
    // d = a * b + a
    // dd/da = b + 1
    // dd/db = a
    const a = new Value(3, [], '', 'a')
    const b = new Value(4, [], '', 'b')
    const prod = a.mul(b) // 12
    const d = prod.add(a) // 12 + 3 = 15

    expect(d.data).toBe(15)

    d.backward()

    expect(d.grad).toBe(1.0)
    expect(b.grad).toBe(3) // a.data
    expect(a.grad).toBe(5) // b.data + 1 = 4 + 1 = 5
  })

  // 10. Diamond / reused-node graph
  it('correctly handles diamond/reused-node graphs', () => {
    // x = a + b (2 + 3 = 5)
    // y = a * b (2 * 3 = 6)
    // z = x * y (5 * 6 = 30)
    // dz/dx = y = 6, dz/dy = x = 5
    // dz/da = (dz/dx)*(dx/da) + (dz/dy)*(dy/da) = 6*1 + 5*b = 6 + 5*3 = 21
    // dz/db = (dz/dx)*(dx/db) + (dz/dy)*(dy/db) = 6*1 + 5*a = 6 + 5*2 = 16
    const a = new Value(2, [], '', 'a')
    const b = new Value(3, [], '', 'b')
    const x = a.add(b)
    const y = a.mul(b)
    const z = x.mul(y)

    expect(z.data).toBe(30)

    z.backward()

    expect(z.grad).toBe(1.0)
    expect(x.grad).toBe(6)
    expect(y.grad).toBe(5)
    expect(a.grad).toBe(21)
    expect(b.grad).toBe(16)
  })

  // 11. Repeated backward() does not double gradients
  it('resets gradients before backward() so repeated calls do not double gradients', () => {
    const a = new Value(3, [], '', 'a')
    const b = new Value(4, [], '', 'b')
    const c = a.mul(b)

    c.backward()
    expect(a.grad).toBe(4)
    expect(b.grad).toBe(3)
    expect(c.grad).toBe(1)

    // Call backward a second time
    c.backward()
    expect(a.grad).toBe(4)
    expect(b.grad).toBe(3)
    expect(c.grad).toBe(1)
  })

  // 12. Topological ordering
  it('computes correct topological ordering with parents before consumers and root last', () => {
    const a = new Value(2, [], '', 'a')
    const b = new Value(3, [], '', 'b')
    const c = a.add(b) // parents: [a, b]
    const d = c.mul(a) // parents: [c, a]

    const order = topologicalOrder(d)

    // Root must be last
    expect(order[order.length - 1]).toBe(d)

    // Each node must appear only once in topologicalOrder
    const uniqueNodes = new Set(order)
    expect(uniqueNodes.size).toBe(order.length)

    // Parents must precede consumers
    const indexOfA = order.indexOf(a)
    const indexOfB = order.indexOf(b)
    const indexOfC = order.indexOf(c)
    const indexOfD = order.indexOf(d)

    expect(indexOfA).toBeLessThan(indexOfC)
    expect(indexOfB).toBeLessThan(indexOfC)
    expect(indexOfC).toBeLessThan(indexOfD)
    expect(indexOfA).toBeLessThan(indexOfD)
  })
})
