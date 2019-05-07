
OPERATIONS = ['+', '-', '*']
CONSTANTS = [str(i) for i in range(10000)]


def remove_redundant_outer_barackets(expr):
    # done not exhaustively
    if len(expr) == 1:
        return expr
    counter = 0
    for i in range(len(expr)):
        if expr[i] == '(':
            counter += 1
        if expr[i] == ')':
            counter -= 1
        if counter == 0 and i < len(expr) - 1:
            break
    else:
        return expr[1:-1]
    return expr


def split_on_first_balanced_symbol(expr):
    expr = remove_redundant_outer_barackets(expr)
    counter = 0
    for i in range(len(expr)):
        if counter == 0 and expr[i] in OPERATIONS:
            symbol = expr[i]
            left_subexpr = expr[:i]
            right_subexpr = expr[i+1:]
            return symbol, left_subexpr, right_subexpr
        if expr[i] == '(':
            counter += 1
        if expr[i] == ')':
            counter -= 1
    assert counter == 0
    raise ValueError('Improper input expression.')


def parser(term):
    '''
    Take expression like (1-((1+3)*2)) and return its parse tree:
    [symbol, left_subtree, right_subtree].
    This function knows nothing about the precedence of operations
    and assumes input term to be exhaustively bracketed and tokenized.
    '''
    term = term.strip(' ')
    if term in CONSTANTS:
        return term
    symbol, left_subexpr, right_subexpr = split_on_first_balanced_symbol(term)
    return [symbol, [parser(left_subexpr), parser(right_subexpr)]]


if __name__ == '__main__':
    print(remove_redundant_outer_barackets('(1-((1+3)*2))'))
    print(remove_redundant_outer_barackets('1-((1+3)*2)'))
    print(remove_redundant_outer_barackets('(1)+2'))
    print(remove_redundant_outer_barackets('(1)'))
    print(remove_redundant_outer_barackets('1'))
    print(split_on_first_balanced_symbol('(1-((1+3)*2))'))
    print(split_on_first_balanced_symbol('1-((1+3)*2)'))

    print(parse('100 -((1+30)* 2)'))
