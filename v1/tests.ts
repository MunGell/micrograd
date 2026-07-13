// run this file with `deno test <file>`

import { assertAlmostEquals, assertEquals, assertStrictEquals, assertThrows } from "jsr:@std/assert";

import Operand from "./operand.ts";
import BackPropagation from "./backpropagation.ts";
import Neuron from "./neuron.ts";
import Layer from "./layer.ts";

// ---------------------------------------------------------------------------
// Operand: construction
// ---------------------------------------------------------------------------

Deno.test("constructor: defaults", () => {
    const a = new Operand(5);
    assertEquals(a.value, 5);
    assertEquals(a.parents, []);
    assertEquals(a.gradient, 0);
    assertEquals(a.operation, "none");
});

Deno.test("constructor: explicit parents and operation", () => {
    const p1 = new Operand(1);
    const p2 = new Operand(2);
    const a = new Operand(3, [p1, p2], "+");
    assertStrictEquals(a.parents[0], p1);
    assertStrictEquals(a.parents[1], p2);
    assertEquals(a.operation, "+");
});

Deno.test("target: sets gradient to 1", () => {
    const a = new Operand(42);
    a.target();
    assertEquals(a.gradient, 1);
});

// ---------------------------------------------------------------------------
// Operand: forward pass
// ---------------------------------------------------------------------------

Deno.test("add: forward value, parents, operation", () => {
    const a = new Operand(2);
    const b = new Operand(3);
    const c = a.add(b);
    assertEquals(c.value, 5);
    assertStrictEquals(c.parents[0], a);
    assertStrictEquals(c.parents[1], b);
    assertEquals(c.operation, "+");
});

Deno.test("subtract: forward value, parents, operation", () => {
    const a = new Operand(2);
    const b = new Operand(3);
    const c = a.subtract(b);
    assertEquals(c.value, -1);
    assertStrictEquals(c.parents[0], a);
    assertStrictEquals(c.parents[1], b);
    assertEquals(c.operation, "-");
});

Deno.test("multiplyBy: forward value, parents, operation", () => {
    const a = new Operand(2);
    const b = new Operand(-3);
    const c = a.multiplyBy(b);
    assertEquals(c.value, -6);
    assertEquals(c.operation, "*");
});

Deno.test("divideBy: forward value, parents, operation", () => {
    const a = new Operand(7);
    const b = new Operand(2);
    const c = a.divideBy(b);
    assertEquals(c.value, 3.5);
    assertEquals(c.operation, "/");
});

Deno.test("divideBy: division by zero yields Infinity (JS semantics)", () => {
    const c = new Operand(5).divideBy(new Operand(0));
    assertEquals(c.value, Infinity);
    const d = new Operand(-5).divideBy(new Operand(0));
    assertEquals(d.value, -Infinity);
});

Deno.test("divideBy: 0/0 yields NaN (JS semantics)", () => {
    const c = new Operand(0).divideBy(new Operand(0));
    assertEquals(Number.isNaN(c.value), true);
});

Deno.test("inPower: forward value, parents, operation", () => {
    const a = new Operand(2);
    const b = new Operand(10);
    const c = a.inPower(b);
    assertEquals(c.value, 1024);
    assertStrictEquals(c.parents[0], a);
    assertStrictEquals(c.parents[1], b);
    assertEquals(c.operation, "**");
});

Deno.test("inPower: fractional and negative exponents", () => {
    assertEquals(new Operand(4).inPower(new Operand(0.5)).value, 2);
    assertEquals(new Operand(2).inPower(new Operand(-1)).value, 0.5);
    assertEquals(new Operand(-2).inPower(new Operand(3)).value, -8);
});

Deno.test("relu: forward for positive, zero, negative", () => {
    assertEquals(new Operand(3).relu().value, 3);
    assertEquals(new Operand(0).relu().value, 0);
    assertEquals(new Operand(-3).relu().value, 0);
    const a = new Operand(-1);
    const r = a.relu();
    assertStrictEquals(r.parents[0], a);
    assertEquals(r.operation, "relu");
});

Deno.test("forward ops do not mutate operands or their gradients", () => {
    const a = new Operand(2);
    const b = new Operand(3);
    a.add(b);
    a.multiplyBy(b);
    assertEquals(a.value, 2);
    assertEquals(b.value, 3);
    assertEquals(a.gradient, 0);
    assertEquals(b.gradient, 0);
});

// ---------------------------------------------------------------------------
// Operand: backprop (single-node local gradients)
// ---------------------------------------------------------------------------

Deno.test("backprop add: both parents receive the gradient", () => {
    const a = new Operand(2);
    const b = new Operand(3);
    const c = a.add(b);
    c.gradient = 4;
    c.backprop();
    assertEquals(a.gradient, 4);
    assertEquals(b.gradient, 4);
});

Deno.test("backprop subtract: minuend +grad, subtrahend -grad", () => {
    const a = new Operand(2);
    const b = new Operand(3);
    const c = a.subtract(b);
    c.gradient = 4;
    c.backprop();
    assertEquals(a.gradient, 4);
    assertEquals(b.gradient, -4);
});

Deno.test("backprop multiply: each parent gets the other's value times grad", () => {
    const a = new Operand(2);
    const b = new Operand(3);
    const c = a.multiplyBy(b);
    c.gradient = 5;
    c.backprop();
    assertEquals(a.gradient, 15); // b.value * grad
    assertEquals(b.gradient, 10); // a.value * grad
});

Deno.test("backprop divide: d(a/b)/da = 1/b, d(a/b)/db = -a/b^2", () => {
    const a = new Operand(6);
    const b = new Operand(3);
    const c = a.divideBy(b);
    c.gradient = 9;
    c.backprop();
    assertAlmostEquals(a.gradient, 9 / 3);        // 3
    assertAlmostEquals(b.gradient, -6 / 9 * 9);   // -6
});

Deno.test("backprop inPower: base gradient d(a^b)/da = b*a^(b-1)", () => {
    const a = new Operand(3);
    const b = new Operand(4);
    const c = a.inPower(b);
    c.gradient = 2;
    c.backprop();
    assertAlmostEquals(a.gradient, 4 * 3 ** 3 * 2); // 216
});

Deno.test("backprop inPower: exponent gradient d(a^b)/db = a^b * ln(a)", () => {
    const a = new Operand(3);
    const b = new Operand(4);
    const c = a.inPower(b);
    c.gradient = 1;
    c.backprop();
    assertAlmostEquals(b.gradient, 3 ** 4 * Math.log(3));
});

Deno.test("backprop inPower: both gradients scale with upstream gradient", () => {
    const a = new Operand(2);
    const b = new Operand(5);
    const c = a.inPower(b); // 32
    c.gradient = 3;
    c.backprop();
    assertAlmostEquals(a.gradient, 5 * 2 ** 4 * 3);          // b*a^(b-1)*grad = 240
    assertAlmostEquals(b.gradient, 32 * Math.log(2) * 3);    // a^b*ln(a)*grad
});

Deno.test("backprop inPower: exponent gradient is 0 for base 1", () => {
    const a = new Operand(1);
    const b = new Operand(7);
    const c = a.inPower(b);
    c.gradient = 1;
    c.backprop();
    assertEquals(b.gradient, 0); // ln(1) = 0
});

Deno.test("backprop inPower: same node as base and exponent (x^x)", () => {
    // d(x^x)/dx = x^x * (ln(x) + 1), reached by accumulating both branches
    const x = new Operand(2);
    const c = x.inPower(x); // 4
    c.gradient = 1;
    c.backprop();
    assertAlmostEquals(x.gradient, 4 * (Math.log(2) + 1));
});

Deno.test("backprop inPower: negative base gives NaN exponent gradient (real ln undefined)", () => {
    const a = new Operand(-2);
    const b = new Operand(3);
    const c = a.inPower(b); // -8
    c.gradient = 1;
    c.backprop();
    assertAlmostEquals(a.gradient, 3 * (-2) ** 2); // base gradient still fine: 12
    assertEquals(Number.isNaN(b.gradient), true);  // ln(-2) is NaN in real arithmetic
});

Deno.test("backprop relu: passes gradient through for positive output", () => {
    const a = new Operand(2);
    const r = a.relu();
    r.gradient = 7;
    r.backprop();
    assertEquals(a.gradient, 7);
});

Deno.test("backprop relu: blocks gradient for negative input", () => {
    const a = new Operand(-2);
    const r = a.relu();
    r.gradient = 7;
    r.backprop();
    assertEquals(a.gradient, 0);
});

Deno.test("backprop relu: zero input gets zero gradient (subgradient convention)", () => {
    const a = new Operand(0);
    const r = a.relu();
    r.gradient = 7;
    r.backprop();
    assertEquals(a.gradient, 0);
});

Deno.test("backprop on a leaf ('none' operation) is a no-op", () => {
    const a = new Operand(5);
    a.gradient = 3;
    a.backprop(); // must not throw, must not touch anything
    assertEquals(a.gradient, 3);
    assertEquals(a.parents, []);
});

Deno.test("backprop accumulates: same node used as both parents", () => {
    const a = new Operand(3);
    const c = a.add(a); // c = 2a
    c.gradient = 1;
    c.backprop();
    assertEquals(a.gradient, 2);

    const b = new Operand(3);
    const d = b.multiplyBy(b); // d = b^2
    d.gradient = 1;
    d.backprop();
    assertEquals(b.gradient, 6); // 2b
});

Deno.test("backprop accumulates: a - a gives zero gradient", () => {
    const a = new Operand(3);
    const c = a.subtract(a);
    c.gradient = 1;
    c.backprop();
    assertEquals(a.gradient, 0);
});

Deno.test("backprop accumulates across multiple calls into existing gradients", () => {
    const a = new Operand(2);
    const b = new Operand(3);
    const c = a.add(b);
    c.gradient = 1;
    c.backprop();
    c.backprop();
    assertEquals(a.gradient, 2);
    assertEquals(b.gradient, 2);
});

// ---------------------------------------------------------------------------
// BackPropagation: whole-graph backward
// ---------------------------------------------------------------------------

Deno.test("backward: single leaf node gets gradient 1", () => {
    const a = new Operand(5);
    BackPropagation.backward(a);
    assertEquals(a.gradient, 1);
});

Deno.test("backward: single addition", () => {
    const a = new Operand(2);
    const b = new Operand(3);
    const c = a.add(b);
    BackPropagation.backward(c);
    assertEquals(c.gradient, 1);
    assertEquals(a.gradient, 1);
    assertEquals(b.gradient, 1);
});

Deno.test("backward: chain g = ((a + x) * t - k) / p", () => {
    const a = new Operand(5);
    const x = new Operand(10);
    const t = new Operand(9);
    const k = new Operand(100);
    const p = new Operand(7);

    const y = a.add(x);        // 15
    const d = y.multiplyBy(t); // 135
    const c = d.subtract(k);   // 35
    const g = c.divideBy(p);   // 5

    BackPropagation.backward(g);

    assertEquals(g.value, 5);
    assertEquals(g.gradient, 1);
    assertAlmostEquals(c.gradient, 1 / 7);
    assertAlmostEquals(p.gradient, -35 / 49);
    assertAlmostEquals(d.gradient, 1 / 7);
    assertAlmostEquals(k.gradient, -1 / 7);
    assertAlmostEquals(y.gradient, 9 / 7);
    assertAlmostEquals(t.gradient, 15 / 7);
    assertAlmostEquals(a.gradient, 9 / 7);
    assertAlmostEquals(x.gradient, 9 / 7);
});

Deno.test("backward: diamond graph e = (a + b) * (a * b)", () => {
    // e = a^2*b + a*b^2  =>  de/da = 2ab + b^2, de/db = a^2 + 2ab
    const a = new Operand(2);
    const b = new Operand(3);
    const c = a.add(b);        // 5
    const d = a.multiplyBy(b); // 6
    const e = c.multiplyBy(d); // 30

    BackPropagation.backward(e);

    assertEquals(e.value, 30);
    assertAlmostEquals(a.gradient, 2 * 2 * 3 + 3 * 3); // 21
    assertAlmostEquals(b.gradient, 2 * 2 + 2 * 2 * 3); // 16
});

Deno.test("backward: node reused as both parents (a * a)", () => {
    const a = new Operand(3);
    const c = a.multiplyBy(a); // a^2
    BackPropagation.backward(c);
    assertEquals(a.gradient, 6);
});

Deno.test("backward: relu inside a graph, active path", () => {
    const a = new Operand(2);
    const b = new Operand(5);
    const r = a.relu();        // 2
    const s = r.multiplyBy(b); // 10
    BackPropagation.backward(s);
    assertEquals(r.gradient, 5);
    assertEquals(a.gradient, 5);
    assertEquals(b.gradient, 2);
});

Deno.test("backward: relu inside a graph, dead path", () => {
    const a = new Operand(-3);
    const b = new Operand(5);
    const r = a.relu();        // 0
    const s = r.multiplyBy(b); // 0
    BackPropagation.backward(s);
    assertEquals(r.gradient, 5);
    assertEquals(a.gradient, 0); // blocked by relu
    assertEquals(b.gradient, 0); // r.value * 1
});

Deno.test("backward: power inside a graph, base and exponent gradients", () => {
    const a = new Operand(2);
    const n = new Operand(3);
    const b = new Operand(4);
    const p = a.inPower(n);    // 8
    const s = p.multiplyBy(b); // 32
    BackPropagation.backward(s);
    assertAlmostEquals(a.gradient, 3 * 2 ** 2 * 4);          // b*a^(n-1) * ds/dp = 48
    assertAlmostEquals(n.gradient, 8 * Math.log(2) * 4);     // a^n*ln(a) * ds/dp
    assertAlmostEquals(b.gradient, 8);
});

Deno.test("backward: division loss shape f = (a / b) * c * g", () => {
    const a = new Operand(5);
    const b = new Operand(10);
    const c = new Operand(70);
    const g = new Operand(5);

    const d = a.divideBy(b);   // 0.5
    const e = d.multiplyBy(c); // 35
    const f = e.multiplyBy(g); // 175

    BackPropagation.backward(f);

    assertEquals(f.gradient, 1);
    assertAlmostEquals(e.gradient, 5);
    assertAlmostEquals(g.gradient, 35);
    assertAlmostEquals(d.gradient, 350);
    assertAlmostEquals(c.gradient, 2.5);
    assertAlmostEquals(a.gradient, 350 / 10);            // 35
    assertAlmostEquals(b.gradient, -350 * 5 / 100);      // -17.5
});

// ---------------------------------------------------------------------------
// Neuron
// ---------------------------------------------------------------------------

Deno.test("Neuron constructor: splits parameters into weights and bias", () => {
    const w1 = new Operand(2);
    const w2 = new Operand(3);
    const b = new Operand(5);
    const n = new Neuron([w1, w2, b]);
    assertEquals(n.weights.length, 2);
    assertStrictEquals(n.weights[0], w1);
    assertStrictEquals(n.weights[1], w2);
    assertStrictEquals(n.bias, b);
    assertEquals(n.parameters.length, 3);
});

Deno.test("Neuron activate: computes w·x + b", () => {
    // 2*10 + 3*20 + 5 = 85
    const n = new Neuron([new Operand(2), new Operand(3), new Operand(5)]);
    const out = n.activate([new Operand(10), new Operand(20)]);
    assertEquals(out.value, 85);
    assertStrictEquals(n.activation, out);
});

Deno.test("Neuron activate: single weight and bias", () => {
    // -4*3 + 2 = -10
    const n = new Neuron([new Operand(-4), new Operand(2)]);
    const out = n.activate([new Operand(3)]);
    assertEquals(out.value, -10);
});

Deno.test("Neuron activate: bias-only neuron accepts empty input", () => {
    const n = new Neuron([new Operand(7)]);
    const out = n.activate([]);
    assertEquals(out.value, 7);
});

Deno.test("Neuron activate: throws on input length mismatch", () => {
    const n = new Neuron([new Operand(2), new Operand(3), new Operand(5)]);
    assertThrows(() => n.activate([new Operand(1)]));
    assertThrows(() => n.activate([new Operand(1), new Operand(2), new Operand(3)]));
});

Deno.test("Neuron activate: gradients flow to weights, inputs, and bias", () => {
    const w1 = new Operand(2);
    const w2 = new Operand(3);
    const b = new Operand(5);
    const x1 = new Operand(10);
    const x2 = new Operand(20);
    const n = new Neuron([w1, w2, b]);
    const out = n.activate([x1, x2]);
    BackPropagation.backward(out);
    assertEquals(w1.gradient, 10); // d(out)/dw_i = x_i
    assertEquals(w2.gradient, 20);
    assertEquals(x1.gradient, 2);  // d(out)/dx_i = w_i
    assertEquals(x2.gradient, 3);
    assertEquals(b.gradient, 1);
});

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

Deno.test("Layer constructor: one neuron per parameter row", () => {
    const params = [
        [new Operand(1), new Operand(2), new Operand(3)],
        [new Operand(4), new Operand(5), new Operand(6)],
    ];
    const layer = new Layer(params);
    assertEquals(layer.neurons.length, 2);
    assertStrictEquals(layer.neurons[0].weights[0], params[0][0]);
    assertStrictEquals(layer.neurons[0].bias, params[0][2]);
    assertStrictEquals(layer.neurons[1].bias, params[1][2]);
});

Deno.test("Layer activate: feeds the same input vector to every neuron", () => {
    // neuron 1: 1*10 + 2*20 + 3 = 53; neuron 2: 4*10 + 5*20 + 6 = 146
    const layer = new Layer([
        [new Operand(1), new Operand(2), new Operand(3)],
        [new Operand(4), new Operand(5), new Operand(6)],
    ]);
    const outs = layer.activate([new Operand(10), new Operand(20)]);
    assertEquals(outs.length, 2);
    assertEquals(outs[0].value, 53);
    assertEquals(outs[1].value, 146);
    assertEquals(layer.activations.length, 2);
});

Deno.test("Layer activate: neuron count independent of input size", () => {
    // 3 neurons over a 2-dimensional input; output width = neuron count
    const layer = new Layer([
        [new Operand(1), new Operand(0), new Operand(0)],
        [new Operand(0), new Operand(1), new Operand(0)],
        [new Operand(1), new Operand(1), new Operand(1)],
    ]);
    const outs = layer.activate([new Operand(4), new Operand(9)]);
    assertEquals(outs.length, 3);
    assertEquals(outs[0].value, 4);
    assertEquals(outs[1].value, 9);
    assertEquals(outs[2].value, 14);
});

Deno.test("Layer activate: throws when input length mismatches weight count", () => {
    const layer = new Layer([[new Operand(1), new Operand(2), new Operand(3)]]); // 2 weights
    assertThrows(() => layer.activate([new Operand(1)])); // 1 input
});

Deno.test("Layer activate: gradients reach only the backpropagated neuron's parameters", () => {
    const w1 = new Operand(1);
    const b1 = new Operand(0);
    const w2 = new Operand(2);
    const b2 = new Operand(0);
    const x = new Operand(5);
    const layer = new Layer([[w1, b1], [w2, b2]]);
    const [o1] = layer.activate([x]);
    BackPropagation.backward(o1);
    assertEquals(w1.gradient, 5);
    assertEquals(b1.gradient, 1);
    assertEquals(w2.gradient, 0); // second neuron untouched by o1's backward
    assertEquals(b2.gradient, 0);
    assertEquals(x.gradient, 1);  // w1.value
});

Deno.test("Layer activate: shared input accumulates gradient across neurons", () => {
    const x = new Operand(5);
    const layer = new Layer([
        [new Operand(1), new Operand(0)],
        [new Operand(2), new Operand(0)],
    ]);
    const [o1, o2] = layer.activate([x]);
    const sum = o1.add(o2);
    BackPropagation.backward(sum);
    assertEquals(x.gradient, 3); // w1 + w2
});
