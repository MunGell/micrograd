import Operand from './operand';

class Neuron {
    parameters: Operand[] = [];
    weigths: Operand[] = [];
    bias: Operand = 0;
    x: Operand[] = [];
    activation: Operand;

    constructor(parameters: Operand[]) {
        this.parameters = parameters;
        this.weights = parameters.slice(0, -1);
        this.bias = parameters.at(-1);
    }

    activate(x: Operand[]) {
        if (x.length !== this.weights.length) {
            throw new Error('weights and input lengths mismatch');
        }
        this.activation = this.weights
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
