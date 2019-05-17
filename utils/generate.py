
import argparse
from random import choice, random


PROBAB = [0.5,1,2,3,4]

def random_arith_term(opers_symb, consts_symb):
    probab = choice(PROBAB)
    def rt(depth=0):
        if random() < probab / 2 ** depth:
            return (choice(opers_symb), (rt(depth + 1), rt(depth + 1)))
        else:
            return choice(consts_symb)
    return rt()

def pretty_print(term, depth=0):
    if len(term) > 1:
        pretty = pretty_print(term[1][0], depth + 1) + \
                            ' ' + term[0] + ' ' + \
                 pretty_print(term[1][1], depth + 1)
        if depth:
            return '( ' + pretty + ' )'
        else:
            return pretty
    else:
        return term[0]


def modulo(expression, modulo=2):
    return eval(expression) % modulo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", default=32, type=int, help="Number of examples to generate.")
    parser.add_argument(
        "--modulo", default=2, type=int, help="Reminder modulo this number.")
    parser.add_argument(
        "--numbers",
        type=str,
        default='0,1,2',
        help="List of numbers for composing expressions. Delimited by ','."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=10,
        help="Maximum length of expression." # TODO tokens?
    )
    parser.add_argument(
        "--operations",
        type=str,
        default='-,+,*',
        help="List of arithmetical operations for composing expressions. \
        Delimited by ','."
    )
    args = parser.parse_args()

numbers = args.numbers.split(',')
operations = args.operations.split(',')

expressions = set()
while len(expressions) < args.n:
    expr = random_arith_term(operations, numbers)
    if len(pretty_print(expr).replace(' ', '')) <= args.max_length:
        expressions.add(expr)

for expr in expressions:
    pretty_expr = pretty_print(expr)
    print(modulo(pretty_expr, args.modulo), '#', pretty_expr)


