import torch
import time
import random
import argparse
import os
import sys
sys.path.append('.') # TODO not like this
from utils.parser import parser


DEVICE = 'cpu'


class FunctionNetwork(torch.nn.Module):
    def __init__(self, function_name, function_arity,
                 dim_in_out=32, dim_h=32):
        super(FunctionNetwork, self).__init__()
        self.function_name = function_name
        self.function_arity = function_arity
        self.model = torch.nn.Sequential(
            torch.nn.Linear(function_arity * dim_in_out, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_in_out),
        ).to(DEVICE)

    def forward(self, x):
        return self.model(x)


class ConstantNetwork(torch.nn.Module):
    def __init__(self, dim_in=10, dim_h=32, dim_out=32):
        super(ConstantNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_out),
        ).to(DEVICE)

    def forward(self, x):
        return self.model(x)


class ModuloNetwork(torch.nn.Module):
    def __init__(self, dim_in=32, dim_h=32, dim_out=2):
        super(ModuloNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_out),
        ).to(DEVICE)

    def forward(self, x):
        return self.model(x)


def read_data(filename):
    with open(filename, 'r') as f:
        data = f.read().splitlines()
    return data


def one_hot(elem, elems):
    if isinstance(elems, int):
        assert 0 <= elem < elems
        elems = range(elems)
    else:
        assert len(set(elems)) == len(elems)
    return [1 if e == elem else 0 for e in elems]


def consts_to_tensors(all_consts):
    return {v: torch.tensor([one_hot(v, all_consts)],
            dtype = torch.float).to(DEVICE) for v in all_consts}


def instanciate_modules(functions, n_constants, modulo):
    modules = {}
    for symb in functions:  # TODO control this
        modules[symb] = FunctionNetwork(symb, functions[symb])
    modules['CONST'] = ConstantNetwork(dim_in=n_constants)
    modules['MODULO'] = ModuloNetwork(dim_out=modulo)
    return modules


def parameters_of_modules(modules):
    parameters = []
    for m in modules:
        parameters.extend(modules[m].parameters())
    return parameters


def tree(term, modules, consts_as_tensors):
    if len(term) > 1:
        x = torch.cat([tree(t, modules, consts_as_tensors) for t in
                       term[1]], -1)
        return modules[term[0]](x)
    else:
        return modules['CONST'](consts_as_tensors[term[0]])


def model(term, parser, modules, consts_as_tensors):
    parsed_term = parser(term)
    return modules['MODULO'](tree(parsed_term, modules, consts_as_tensors))


def loss(outputs, targets):
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    return criterion(outputs, targets)


def train(inputs, labels, modules, parser, loss, optimizer, epochs=1):
    def _model(term): return model(term, parser, modules, consts_as_tensors)
    assert len(inputs) == len(labels)
    for e in range(epochs):
        ls = []
        ps = []
        for i in range(len(inputs)):
            p = _model(inputs[i])
            l = loss(p, labels[i])
            ls.append(l.item())
            ps.append(p.argmax().item())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        l_avg = sum(ls) / len(ls)
        return l_avg
        # acc = sum(ps[i] == labels[i].item() \
        #          for i in range(len(labels))) / len(labels)
        #print("Loss on training {}. Accuracy on training {}.".format(l_avg, acc))


def predict(inputs, model):
    return [model(i).argmax().item() for i in inputs]


def accuracy(inputs, labels, modules):
    def _model(term): return model(term, parser, modules, consts_as_tensors)
    preds = predict(inputs, _model)
    return sum(preds[i] == labels[i].item()
               for i in range(len(labels))) / len(labels)


############ TEST ###############################################


args_parser = argparse.ArgumentParser()
args_parser.add_argument(
    "--data_dir",
    type=str,
    help="Path to training and validation data."
    "It's assumed this dir contains files:"
    "train.in, train.out, valid.in, valid.out")
args_parser.add_argument(
    "--model_dir",
    default='',
    type=str,
    help="Path where to save the trained model.")
args_parser.add_argument(
    "--epochs",
    default=10,
    type=int,
    help="Number of epochs.")
args_parser.add_argument(
    "--embed_dim",
    default=8,
    type=int,
    help="Token embedding dimension.")
args = args_parser.parse_args()



FUNCTIONS_WITH_ARITIES = {
    '+': 2,
    '*': 2,
    '-': 2
}


inputs_train = read_data(os.path.join(args.data_dir, 'train.in'))
labels_train = read_data(os.path.join(args.data_dir, 'train.out'))
inputs_valid = read_data(os.path.join(args.data_dir, 'valid.in'))
labels_valid = read_data(os.path.join(args.data_dir, 'valid.out'))
inputs_vocab = read_data(os.path.join(args.data_dir, 'vocab.in'))
labels_vocab = read_data(os.path.join(args.data_dir, 'vocab.out'))

labels_train = [torch.tensor([int(l)]).to('cpu') for l in labels_train]
labels_valid = [torch.tensor([int(l)]).to('cpu') for l in labels_valid]

modulo = len(labels_vocab) + 1
constants = set(inputs_vocab) - set(FUNCTIONS_WITH_ARITIES) - {'(', ')'}
consts_as_tensors = consts_to_tensors(constants)
modules = instanciate_modules(FUNCTIONS_WITH_ARITIES, len(constants), modulo)
params_of_modules = parameters_of_modules(modules)
loss_1 = loss
optim_1 = torch.optim.SGD(params_of_modules, lr=0.001, momentum=0.8)
for e in range(args.epochs):
    #t0 = time.time()
    print(f"\nEpoch {e}.")
    loss_train = train(inputs_train, labels_train, modules, parser, loss_1, optim_1)
    print(f"Loss on training: {loss_train}")
    acc_train = accuracy(inputs_train, labels_train, modules)
    acc_valid = accuracy(inputs_valid, labels_valid, modules)
    print(f"Accuracy on training: {acc_train}")
    print(f"Accuracy on validation: {acc_valid}")
    #print('Time:', time.time() - t0)
