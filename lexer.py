from enum import Enum
from dataclasses import dataclass
from typing import List, Union, Callable, Dict
import re
import sys
import operator

class TokenType(Enum):
    PLUS = 1
    MINUS = 2
    TIMES = 3
    DIVIDE = 4
    PRINT = 5
    NUMBER = 6
    DUP = 7   # ( a -- a a )
    DROP = 8  # ( a -- )
    SWAP = 9  # ( a b -- b a )
    OVER = 10 # ( a b -- a b a )
    ROT = 11  # ( a b c -- b c a )
    PRINTALL = 12


@dataclass
class Token:
    type: TokenType
    value: object

OPS = {
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.TIMES,
    "/": TokenType.DIVIDE,
    "print": TokenType.PRINT,
    "dup": TokenType.DUP,
    "drop": TokenType.DROP,
    "swap": TokenType.SWAP,
    "over": TokenType.OVER,
    "rot": TokenType.ROT,
    "printall": TokenType.PRINTALL,
}

# int or float, optional leading sign:
# 5, -4, +7, 2.1, -3.0, .5, 5.
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")

def tokenize(source_code: str) -> List[Token]:
    tokens: List[Token] = []
    parts = source_code.split()

    for p in parts:
        if p in OPS:
            tokens.append(Token(OPS[p], p))
            continue

        if NUMBER_RE.match(p):
            # decide int vs float by presence of '.'
            val: Union[int, float]
            if "." in p:
                val = float(p)
            else:
                val = int(p)
            tokens.append(Token(TokenType.NUMBER, val))
            continue

        raise ValueError(f"Unexpected token: {p!r}")

    return tokens

def pop2(stack):
    if len(stack) < 2:
        raise ValueError(f"Need 2 values, stack has {len(stack)}: {stack}")
    b = stack.pop()
    a = stack.pop()
    return a, b

def need(stack, n: int, opname: str):
    if len(stack) < n:
        raise ValueError(f"{opname} needs {n} value(s), stack has {len(stack)}: {stack}")

def make_binop(fn, name: str):
    def op(stack):
        a, b = pop2(stack, name)
        stack.append(fn(a, b))
    return op

def op_div(stack):
    a, b = pop2(stack, "/")
    if b == 0:
        raise ValueError("Division by zero")
    stack.append(a / b)

def op_print(stack):
    need(stack, 1, "print")
    print(stack.pop())

def op_dup(stack):
    need(stack, 1, "dup")
    stack.append(stack[-1])

def op_drop(stack):
    need(stack, 1, "drop")
    stack.pop()

def op_swap(stack):
    need(stack, 2, "swap")
    stack[-1], stack[-2] = stack[-2], stack[-1]

def op_over(stack):
    need(stack, 2, "over")
    stack.append(stack[-2])

def op_rot(stack):
    need(stack, 3, "rot")
    # (a b c -- b c a)
    a, b, c = stack[-3], stack[-2], stack[-1]
    stack[-3], stack[-2], stack[-1] = b, c, a

def op_printall(stack):
    print(stack)

DISPATCH: Dict[TokenType, Callable[[list], None]] = {
    TokenType.PLUS: make_binop(operator.add, "+"),
    TokenType.MINUS: make_binop(operator.sub, "-"),
    TokenType.TIMES: make_binop(operator.mul, "*"),
    TokenType.DIVIDE: op_div,
    TokenType.PRINT: op_print,
    TokenType.DUP: op_dup,
    TokenType.DROP: op_drop,
    TokenType.SWAP: op_swap,
    TokenType.OVER: op_over,
    TokenType.ROT: op_rot,
    TokenType.PRINTALL: op_printall,
}

def evaluate_tokens(tokens: List[Token]):
    stack = []

    for tok in tokens:
        if tok.type == TokenType.NUMBER:
            stack.append(tok.value)
            continue

        op = DISPATCH.get(tok.type)
        if op is None:
            raise ValueError(f"Unhandled token type: {tok.type}")

        op(stack)

    return stack

if __name__ == "__main__":
    source = sys.argv[1]
    with open(source, encoding="utf-8") as f:
        tokens = tokenize(f.read())
    evaluate_tokens(tokens)