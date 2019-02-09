from torch import optim


def optimizer_class(name):
    if name == 'sgd':
        OptimizerClass = optim.SGD
    elif name == 'adagrad':
        OptimizerClass = optim.Adagrad
    elif name == 'adadelta':
        OptimizerClass = optim.Adadelta
    elif name == 'adam':
        OptimizerClass = optim.Adam
    elif name == 'sparseadam':
        OptimizerClass = optim.SparseAdam
    else:
        raise RuntimeError("Invalid optim method: " + name)
    return OptimizerClass
