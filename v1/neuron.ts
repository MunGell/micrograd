import Operand from './operand';

class Neuron {
    weigths: Operand[] = [];
    bias: Operand = 0;
    x: Operand[] = [];
    activation: Operand;

    constructor(weights: Operand[], bias: Operand) {
        this.weights = weights;
        this.bias = bias;
    }

    activate(x: Operand[]) {
        if (x.length !== this.weights.length) {
            throw new Error('weights and input lengths mismatch');
        }
        this.activation = this.weight
            .map((w, i) => w.multiplyBy(x[i])) // map multiplies weight with input
            .reduce((a, v) => a.add(v),0)      // then reduce sums it all up
            .add(this.bias);                   // then we add bias
        return this;
    }

    relu() {
        return this.activation.relu();
    }
}

export default Neuron;
