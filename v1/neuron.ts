import Operand from './operand.ts';

class Neuron {
    parameters: Operand[];
    weights: Operand[];
    bias: Operand;
    activation?: Operand;

    constructor(parameters: Operand[]) {
        this.parameters = parameters;
        this.weights = parameters.slice(0, -1);
        this.bias = parameters.at(-1)!;
    }

    activate(input: Operand[]) {
        if (input.length !== this.weights.length) {
            throw new Error('weights and input lengths mismatch');
        }
        this.activation = this.weights
            .map((w, i) => w.multiplyBy(input[i]))       // multiplies weights with inputs
            .reduce((a, v) => a.add(v), new Operand(0))  // then sum it all up
            .add(this.bias);                             // then add bias

        return this.activation;
    }
}

export default Neuron;
