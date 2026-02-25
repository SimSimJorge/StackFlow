from enum import Enum
from dataclasses import dataclass
from typing import List, Union
import re
import sys

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

def evaluate_tokens(tokens):
    stack = []
    for token in tokens:
        t = token.type

        if t == TokenType.NUMBER:
            stack.append(token.value)
        elif t == TokenType.PLUS:
            a, b = pop2(stack)
            stack.append(a + b)
        elif t == TokenType.MINUS:
            a, b = pop2(stack)
            stack.append(a - b)
        elif t == TokenType.TIMES:
            a, b = pop2(stack)
            stack.append(a * b)
        elif t == TokenType.DIVIDE:
            a, b = pop2(stack)
            if b==0:
                raise ValueError(f"Division by zero")
            stack.append(a / b)
        elif t == TokenType.PRINT:
            print(stack.pop())
        elif t == TokenType.DUP:
            need(stack, 1, "dup")
            stack.append(stack[-1])
        elif t == TokenType.DROP:
            need(stack, 1, "drop")
            stack.pop()
        elif t == TokenType.SWAP:
            need(stack, 2, "swap")
            stack[-1], stack[-2] = stack[-2], stack[-1]
        elif t == TokenType.OVER:
            need(stack, 2, "over")
            stack.append(stack[-2])
        elif t == TokenType.ROT:
            need(stack, 3, "rot")
            stack[-1], stack[-2], stack[-3] = stack[-3], stack[-1], stack[-2]
        elif t == TokenType.PRINTALL:
            print(stack)

    return stack

if __name__ == '__main__':
    source = sys.argv[1]
    with open(source) as f:
        tokens = tokenize(f.read())
        #print(tokens)

    evaluate_tokens(tokens)