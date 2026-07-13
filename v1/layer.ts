import Operand from './operand.ts';
import Neuron from './neuron.ts';

class Layer {
    parameters: Operand[][] = [[]];
    neurons: Neuron[] = [];
    activations: Operand[] = [];

    constructor(parameters: Operand[][]) {
        this.parameters = parameters;
        this.neurons = parameters.map(p => new Neuron(p));
    }

    activate(input: Operand[]) {
        this.activations = this.neurons.map(n => n.activate(input));

        return this.activations;
    }
}

export default Layer;
