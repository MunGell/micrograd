import Operand from "./operand.ts";

class BackPropagation {
    static backward(start: Operand) {
        const topologicalSort: Operand[] = [];
        const visited = new Set<Operand>();

        function buildTopology(node: Operand) {
            if (!visited.has(node)) {
                visited.add(node);
                for (const parent of node.parents) {
                    buildTopology(parent);
                }
                topologicalSort.push(node);
            }
        }

        buildTopology(start);

        start.target();

        for (let i = topologicalSort.length - 1; i >= 0; i--) {
            topologicalSort[i].backprop();
        }
    }
}

export default BackPropagation;
