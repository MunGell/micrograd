import Operand from './operand.ts';
import Layer from './layer.ts';

class MultiLayerPerceptron {
    layers: Layer[];

    constructor(weights) {
        this.layers = weights.map(layerData => new Layer(
            layerData.neurons.map(neuron => neuron.map(weight => new Operand(weight))),
            layerData.mode
        ));
    }

    predict(input: Number[]) {
        let nextInput = input.map(number => new Operand(number));

        this.layers.forEach(layer => {
            nextInput = layer.activate(nextInput);
        });

        return nextInput;
    }
}

export default MultiLayerPerceptron;
