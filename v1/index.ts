import modelWeights from './data/mlp.json';
import MLP from './multilayer-perceptron.ts'

const model = new MLP(modelWeights.layers);

model.predict()
