class Operand {
    value: number;
    parents: Operand[];
    gradient: number = 0;
    operation: string;


    constructor(value: number, parents: Operand[] = [], operation: string = 'none') {
        this.value = value;
        this.parents = parents;
        this.operation = operation;
    }

    add(other: Operand) {
        return new Operand(this.value + other.value, [this, other], '+');
    }

    subtract(other: Operand) {
        return new Operand(this.value - other.value, [this, other], '-');
    }

    multiplyBy(other: Operand) {
        return new Operand(this.value * other.value, [this, other], '*');
    }

    divideBy(other: Operand) {
        return new Operand(this.value / other.value, [this, other], '/');
    }

    inPower(power: Operand) {
        return new Operand(this.value ** power.value, [this, power], '**');
    }

    relu() {
        return new Operand(this.value < 0 ? 0 : this.value, [this], 'relu');
    }

    target() {
        this.gradient = 1;
    }

    backprop() {
        switch (this.operation) {
            case '+':
                this.parents[0].gradient += this.gradient;
                this.parents[1].gradient += this.gradient;
                break;
            case '-':
                this.parents[0].gradient += this.gradient;
                this.parents[1].gradient -= this.gradient;
                break;
            case '*':
                this.parents[0].gradient += this.parents[1].value * this.gradient;
                this.parents[1].gradient += this.parents[0].value * this.gradient;
                break;
            case '/':
                this.parents[0].gradient += this.parents[1].value ** -1 * this.gradient;
                this.parents[1].gradient += -1 * this.parents[0].value * this.parents[1].value ** -2 * this.gradient;
                break;
            case '**':
                this.parents[0].gradient += this.parents[1].value * this.parents[0].value ** (this.parents[1].value - 1) * this.gradient;
                this.parents[1].gradient += this.value * Math.log(this.parents[0].value) * this.gradient;
                break;
            case 'relu':
                this.parents[0].gradient += (this.value > 0 ? 1 : 0) * this.gradient;
        }
    }
}

export default Operand;
