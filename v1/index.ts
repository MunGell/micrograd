import modelWeights from './data/linear-regression.json' with { type: "json" };
import MLP from './multilayer-perceptron.ts'

const model = new MLP(modelWeights.layers);

console.log(model.predict([1])[0].value);
