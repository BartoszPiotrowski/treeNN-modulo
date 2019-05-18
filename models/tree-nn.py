import torch
import time
import random
import argparse
import os
import sys
import logging
sys.path.append('.') # TODO not like this
from utils.parser import parser


DEVICE = 'cpu'


def MLP(num_layers, dim_in, dim_hid, dim_out):
    def lin_relu(dim_in, dim_out):
        return [
            torch.nn.Linear(dim_in, dim_out),
            torch.nn.ReLU()
        ]
    if num_layers == 1:
        return torch.nn.Sequential(*lin_relu(dim_in, dim_out)).to(DEVICE)
    else:
        return torch.nn.Sequential(
            *lin_relu(dim_in, dim_hid) + \
            lin_relu(dim_hid, dim_hid) * (num_layers - 2) + \
            lin_relu(dim_hid, dim_out)
        ).to(DEVICE)


class FunctionNetwork(torch.nn.Module):
    def __init__(self, function_name, function_arity, num_layers, num_units):
        super(FunctionNetwork, self).__init__()
        self.function_name = function_name
        self.function_arity = function_arity
        #self.model = MLP(num_layers, num_units * function_arity, num_units, num_units)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(function_arity * 32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
        ).to(DEVICE)

    def forward(self, x):
        return self.model(x)


class ConstantNetwork(torch.nn.Module):
    def __init__(self, num_layers, num_constants, num_units):
        super(ConstantNetwork, self).__init__()
        #self.model = MLP(num_layers, num_units_in, num_units, num_units)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_constants, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
        ).to(DEVICE)

    def forward(self, x):
        return self.model(x)


class ModuloNetwork(torch.nn.Module):
    def __init__(self, num_layers, num_units, modulo):
        super(ModuloNetwork, self).__init__()
        #self.model = MLP(num_layers, num_units, num_units, modulo)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, modulo),
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


def instanciate_modules(functions, num_constants, modulo, num_layers, num_units):
    modules = {}
    for symb in functions:  # TODO control this
        modules[symb] = FunctionNetwork(
            symb, functions[symb], num_layers, num_units)
    modules['CONST'] = ConstantNetwork(num_layers, num_constants, num_units)
    modules['MODULO'] = ModuloNetwork(num_layers, num_units, modulo)
    logging.info(modules)
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


def train(inputs, labels, modules, consts_as_tensors, parser, loss, optimizer):
    #train one epoch
    assert len(inputs) == len(labels)
    ls = []
    ps = []
    for i in range(len(inputs)):
        p = model(inputs[i], parser, modules, consts_as_tensors)
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
    #logging.info("Loss on training {}. Accuracy on training {}.".format(l_avg, acc))


def predict(inputs, model):
    return [model(i).argmax().item() for i in inputs]


def accuracy(inputs, labels, modules):
    _model = lambda term: model(term, parser, modules, consts_as_tensors)
    preds = predict(inputs, _model)
    return sum(preds[i] == labels[i].item()
               for i in range(len(labels))) / len(labels)

def difficulty_doser(inputs, prev_level=0, levels=10):
    # returns indices of inputs in length-dependent partitions
    # 1, 2, ..., prev_level + 1
    if prev_level >= levels:
        return False
    lengths = [len(i.split(' ')) for i in inputs]
    limit_index = int((prev_level + 1) * len(inputs) / levels)
    sorted_indices = [i[0] for i in sorted(enumerate(lengths), key=lambda x:x[1])]
    return sorted_indices[:limit_index]




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
    "--num_layers",
    default=2,
    type=int,
    help="Number of layers in MLP modules for functions, constants and modulo.")
args_parser.add_argument(
    "--num_units",
    default=32,
    type=int,
    help="Number of units in layers in MLP modules for functions, constants and modulo.")
args_parser.add_argument(
    "--lr",
    default=0.001,
    type=float,
    help="Learning rate.")
args_parser.add_argument(
    "--momentum",
    default=0.8,
    type=int,
    help="Momentum rate.")
args_parser.add_argument(
    "--short_examples_first",
    action='store_true',
    help="Train first on short examples.")
args_parser.add_argument(
    "--log_file",
    default='',
    type=str,
    help="Name of a log file.")
args = args_parser.parse_args()


FUNCTIONS_WITH_ARITIES = {
    '+': 2,
    '*': 2,
    '-': 2
}


logging.basicConfig(filename=args.log_file,
                    level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

inputs_train = read_data(os.path.join(args.data_dir, 'train.in'))
labels_train = read_data(os.path.join(args.data_dir, 'train.out'))
inputs_valid = read_data(os.path.join(args.data_dir, 'valid.in'))
labels_valid = read_data(os.path.join(args.data_dir, 'valid.out'))
inputs_vocab = read_data(os.path.join(args.data_dir, 'vocab.in'))
labels_vocab = read_data(os.path.join(args.data_dir, 'vocab.out'))

labels_train = [torch.tensor([int(l)]).to(DEVICE) for l in labels_train]
labels_valid = [torch.tensor([int(l)]).to(DEVICE) for l in labels_valid]

modulo = len(labels_vocab)
constants = set(inputs_vocab) - set(FUNCTIONS_WITH_ARITIES) - {'(', ')'}
consts_as_tensors = consts_to_tensors(constants)
modules = instanciate_modules(FUNCTIONS_WITH_ARITIES, len(constants), modulo,
                              args.num_layers, args.num_units)
params_of_modules = parameters_of_modules(modules)
loss_1 = loss
optim_1 = torch.optim.SGD(params_of_modules, args.lr, args.momentum)

if not args.short_examples_first:
    for e in range(args.epochs):
        #t0 = time.time()
        logging.info(f"\nEpoch {e}.")
        loss_train = train(inputs_train, labels_train,
                           modules, consts_as_tensors, parser, loss_1, optim_1)
        logging.info(f"Loss on training: {loss_train}")
        #acc_train = accuracy(inputs_train, labels_train, modules)
        acc_valid = accuracy(inputs_valid, labels_valid, modules)
        #logging.info(f"Accuracy on training: {acc_train}")
        logging.info(f"Accuracy on validation: {acc_valid}")
        #logging.info('Time:', time.time() - t0)
else:
    level = 0
    indices_current_level = difficulty_doser(inputs_train, level)
    inputs_train_current_level = [inputs_train[i] for i in indices_current_level]
    labels_train_current_level = [labels_train[i] for i in indices_current_level]
    max_l = max([len(i.split(' ')) for i in inputs_train_current_level])
    for e in range(args.epochs):
        logging.info(f"\nEpoch {e}. Level {level}. Max length of examples {max_l}.")
        logging.info(inputs_train_current_level[-1])
        loss_train = train(inputs_train_current_level, labels_train_current_level,
                           modules, consts_as_tensors, parser, loss_1, optim_1)
        logging.info(f"Loss on training: {loss_train}")
        acc_train = accuracy(inputs_train_current_level, labels_train_current_level, modules)
        logging.info(f"Accuracy on current training subset: {acc_train}")
        acc_valid = accuracy(inputs_valid, labels_valid, modules)
        logging.info(f"Accuracy on validation: {acc_valid}")
        if acc_train > 0.8:
            level = level + 1
            try:
                indices_current_level = difficulty_doser(inputs_train, level)
                inputs_train_current_level = [inputs_train[i] for i in indices_current_level]
                labels_train_current_level = [labels_train[i] for i in indices_current_level]
                max_l = max([len(i.split(' ')) for i in inputs_train_current_level])
            except:
                logging.info("Trining finished.")
                break
