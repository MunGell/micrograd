import Operand from './operand.ts';
import Neuron from './neuron.ts';

export type LayerMode = 'raw' | 'relu';

class Layer {
    parameters: Operand[][] = [[]];
    neurons: Neuron[] = [];
    activations: Operand[] = [];
    mode: LayerMode = 'raw';

    constructor(parameters: Operand[][], mode: LayerMode = 'raw') {
        this.parameters = parameters;
        this.neurons = parameters.map(p => new Neuron(p));
        this.mode = mode;
    }

    activate(input: Operand[]) {
        this.activations = this.neurons.map(n => n.activate(input));

        if (this.mode == 'relu') {
            this.activations = this.activations.map(a => a.relu());
        }

        return this.activations;
    }
}

export default Layer;
