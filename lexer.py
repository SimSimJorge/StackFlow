from enum import Enum
from dataclasses import dataclass
from typing import List, Union, Callable, Dict
import re
import sys
import operator


# ============================================================
# Token Definitions
# ============================================================

class TokenType(Enum):
    """
    Enumeration of all supported token types in the language.
    These represent operations and literal values.
    """
    PLUS = 1
    MINUS = 2
    TIMES = 3
    DIVIDE = 4
    PRINT = 5
    NUMBER = 6

    # Stack manipulation operations (Forth-style)
    DUP = 7   # ( a -- a a )
    DROP = 8  # ( a -- )
    SWAP = 9  # ( a b -- b a )
    OVER = 10 # ( a b -- a b a )
    ROT = 11  # ( a b c -- b c a )

    PRINTALL = 12  # Print entire stack


@dataclass
class Token:
    """
    Represents a lexical token produced by the tokenizer.
    type  -> TokenType (operation or NUMBER)
    value -> Raw value (string for ops, int/float for numbers)
    """
    type: TokenType
    value: object


# ============================================================
# Operator Mapping
# ============================================================

# Maps string representations to TokenType
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


# ============================================================
# Number Recognition
# ============================================================

# Matches:
# 5, -4, +7, 2.1, -3.0, .5, 5.
# Supports optional leading sign and decimal formats
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


# ============================================================
# Tokenizer
# ============================================================

def tokenize(source_code: str) -> List[Token]:
    """
    Converts source code string into a list of Token objects.

    Splits by whitespace and classifies each part as:
    - Operator
    - Number (int or float)
    - Error (invalid token)
    """
    tokens: List[Token] = []
    parts = source_code.split()

    for p in parts:

        # Check if token is a known operator
        if p in OPS:
            tokens.append(Token(OPS[p], p))
            continue

        # Check if token is a valid number
        if NUMBER_RE.match(p):
            # Decide between int and float based on presence of '.'
            val: Union[int, float]
            if "." in p:
                val = float(p)
            else:
                val = int(p)

            tokens.append(Token(TokenType.NUMBER, val))
            continue

        # If it matches neither operator nor number → error
        raise ValueError(f"Unexpected token: {p!r}")

    return tokens


# ============================================================
# Stack Utility Helpers
# ============================================================

def pop2(stack):
    """
    Pops two values from the stack (a, b).
    Raises error if fewer than 2 values exist.
    """
    if len(stack) < 2:
        raise ValueError(f"Need 2 values, stack has {len(stack)}: {stack}")
    b = stack.pop()
    a = stack.pop()
    return a, b


def need(stack, n: int, opname: str):
    """
    Ensures stack has at least n elements.
    Used for validating stack operations.
    """
    if len(stack) < n:
        raise ValueError(f"{opname} needs {n} value(s), stack has {len(stack)}: {stack}")


# ============================================================
# Binary Operator Factory
# ============================================================

def make_binop(fn, name: str):
    """
    Creates a binary stack operation.

    Pops two values (a, b) and pushes fn(a, b).
    Used to generate +, -, * operations dynamically.
    """
    def op(stack):
        a, b = pop2(stack)
        stack.append(fn(a, b))
    return op


# ============================================================
# Operation Implementations
# ============================================================

def op_div(stack):
    """Division operation with zero-division protection."""
    a, b = pop2(stack)
    if b == 0:
        raise ValueError("Division by zero")
    stack.append(a / b)


def op_print(stack):
    """Pops and prints top of stack."""
    need(stack, 1, "print")
    print(stack.pop())


def op_dup(stack):
    """Duplicates top value of stack. (a -- a a)"""
    need(stack, 1, "dup")
    stack.append(stack[-1])


def op_drop(stack):
    """Removes top value of stack. (a -- )"""
    need(stack, 1, "drop")
    stack.pop()


def op_swap(stack):
    """Swaps top two values. (a b -- b a)"""
    need(stack, 2, "swap")
    stack[-1], stack[-2] = stack[-2], stack[-1]


def op_over(stack):
    """Copies second value to top. (a b -- a b a)"""
    need(stack, 2, "over")
    stack.append(stack[-2])


def op_rot(stack):
    """Rotates top three values. (a b c -- b c a)"""
    need(stack, 3, "rot")
    a, b, c = stack[-3], stack[-2], stack[-1]
    stack[-3], stack[-2], stack[-1] = b, c, a


def op_printall(stack):
    """Prints entire stack without modifying it."""
    print(stack)


# ============================================================
# Dispatch Table
# ============================================================

# Maps TokenType to the corresponding operation function
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


# ============================================================
# Evaluator
# ============================================================

def evaluate_tokens(tokens: List[Token]):
    """
    Evaluates a sequence of tokens using a stack-based execution model.

    - Numbers are pushed onto the stack.
    - Operations manipulate the stack via the dispatch table.
    """
    stack = []

    for tok in tokens:

        # Push numbers directly
        if tok.type == TokenType.NUMBER:
            stack.append(tok.value)
            continue

        # Lookup operation
        op = DISPATCH.get(tok.type)
        if op is None:
            raise ValueError(f"Unhandled token type: {tok.type}")

        # Execute operation
        op(stack)

    return stack


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    """
    Program entry point.

    Usage:
        python program.py <sourcefile>

    Reads source file, tokenizes it, and evaluates it.
    """
    source = sys.argv[1]

    with open(source, encoding="utf-8") as f:
        tokens = tokenize(f.read())

    evaluate_tokens(tokens)