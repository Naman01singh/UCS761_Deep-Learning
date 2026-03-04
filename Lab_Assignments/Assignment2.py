#AND Gate
def perceptron_AND(x1, x2):
    w1 = 1
    w2 = 1
    bias = -1.5
    net = (w1 * x1) + (w2 * x2) + bias
    if(net >= 0):
        return 1
    return 0

print("AND Gate")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(x1, x2, "-", perceptron_AND(x1, x2))

# OR Gate
def perceptron_OR(x1, x2):
    w1 = 1
    w2 = 1
    bias = -0.5
    net = (w1 * x1) + (w2 * x2) + bias
    if(net >= 0):
        return 1
    return 0

print("OR Gate")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(x1, x2, "-", perceptron_OR(x1, x2))

# NAND Gate
def perceptron_NAND(x1, x2):
    w1 = -1
    w2 = -1
    bias = 1.5
    net = (w1 * x1) + (w2 * x2) + bias
    if(net >= 0):
        return 1
    return 0

print("NAND Gate")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(x1, x2, "-", perceptron_NAND(x1, x2))

# NOR Gate
def perceptron_NOR(x1, x2):
    w1 = -1
    w2 = -1
    bias = 0.5
    net = (w1 * x1) + (w2 * x2) + bias
    if(net >= 0):
        return 1
    return 0

print("NOR Gate")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(x1, x2, "-", perceptron_NOR(x1, x2))