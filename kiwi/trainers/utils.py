from torch import optim


def OptimizerClass(name):
    if name == 'sgd':
        optimizer = optim.SGD
    elif name == 'adagrad':
        optimizer = optim.Adagrad
    elif name == 'adadelta':
        optimizer = optim.Adadelta
    elif name == 'adam':
        optimizer = optim.Adam
    elif name == 'sparseadam':
        optimizer = optim.SparseAdam
    else:
        raise RuntimeError("Invalid optim method: " + name)
    return optimizer
