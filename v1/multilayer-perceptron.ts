import Operand from './operand.ts';
import Layer, { LayerMode } from './layer.ts';

type LayerData = {
    mode: string
    neurons: number[][]
}

class MultiLayerPerceptron {
    layers: Layer[];

    constructor(weights: LayerData[]) {
        this.layers = weights.map(layerData => new Layer(
            layerData.neurons.map(neuron => neuron.map(weight => new Operand(weight))),
            this._toLayerMode(layerData.mode)
        ));
    }

    predict(input: number[]) {
        let nextInput = input.map(number => new Operand(number));

        this.layers.forEach(layer => {
            nextInput = layer.activate(nextInput);
        });

        return nextInput;
    }

    _toLayerMode(mode: string): LayerMode {
        if (mode === 'raw' || mode === 'relu') return mode;
        throw new Error(`unknown layer mode: ${mode}`);
    }
}

export default MultiLayerPerceptron;
