"""
Core numerical methods and helpers for the Zero of Functions (ZOF) project.

This module centralizes:
    - Parsing user-provided expressions into safe callables.
    - Implementations of six classical root-finding algorithms.
    - A thin façade (`run_method`) that normalizes inputs/outputs so both the
      CLI and Flask web layers can consume the same API.

Each solver returns a dictionary with:
    converged: bool
    root: Optional[float]
    final_error: Optional[float]
    iterations: List[Dict[str, float]]
    columns: List[str]   # keys to display for each iteration dict
    message: str
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import math
import sympy as sp


# --------------------------------------------------------------------------- #
# Expression parsing helpers
# --------------------------------------------------------------------------- #

X_SYMBOL = sp.symbols("x")


class ExpressionParseError(ValueError):
    """Raised when a user-supplied expression cannot be parsed."""


def _sympy_to_float(value: float) -> float:
    """Convert sympy/complex numbers into a safe float."""
    if isinstance(value, complex):
        if abs(value.imag) > 1e-9:
            raise ValueError("Expression evaluated to a complex number.")
        return float(value.real)
    try:
        return float(value)
    except TypeError as exc:  # pragma: no cover - defensive (sympy edge cases)
        raise ValueError("Unable to convert expression result to float.") from exc


def build_function(expr: str) -> Callable[[float], float]:
    """
    Convert an input string into a callable f(x).

    Users can enter expressions such as:
        "x**3 - 5*x + 2", "sin(x) - x/2", etc.
    """
    if not expr or not expr.strip():
        raise ExpressionParseError("Function expression cannot be empty.")
    try:
        sympy_expr = sp.sympify(expr, locals={"x": X_SYMBOL})
        func = sp.lambdify(X_SYMBOL, sympy_expr, "math")
    except (sp.SympifyError, SyntaxError) as exc:
        raise ExpressionParseError(f"Invalid function expression: {expr}") from exc

    def wrapper(value: float) -> float:
        try:
            evaluated = func(value)
        except Exception as exc:  # pragma: no cover - runtime math errors
            raise ValueError(f"Error evaluating function at x={value}: {exc}") from exc
        return _sympy_to_float(evaluated)

    return wrapper


def build_derivative(expr: str) -> Callable[[float], float]:
    """Automatically differentiate the expression for Newton-based methods."""
    try:
        sympy_expr = sp.sympify(expr, locals={"x": X_SYMBOL})
    except (sp.SympifyError, SyntaxError) as exc:
        raise ExpressionParseError(
            "Original expression is invalid; cannot build derivative."
        ) from exc
    derivative = sp.diff(sympy_expr, X_SYMBOL)
    func = sp.lambdify(X_SYMBOL, derivative, "math")

    def wrapper(value: float) -> float:
        evaluated = func(value)
        return _sympy_to_float(evaluated)

    return wrapper


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass
class MethodResult:
    converged: bool
    root: Optional[float]
    final_error: Optional[float]
    iterations: List[Dict[str, float]]
    columns: List[str]
    message: str


def _base_result(columns: List[str]) -> MethodResult:
    return MethodResult(
        converged=False,
        root=None,
        final_error=None,
        iterations=[],
        columns=columns,
        message="",
    )


def _finalize(
    result: MethodResult,
    *,
    converged: bool,
    root: Optional[float],
    error: Optional[float],
    message: str,
) -> MethodResult:
    result.converged = converged
    result.root = root
    result.final_error = error
    result.message = message
    return result


# --------------------------------------------------------------------------- #
# Numerical methods
# --------------------------------------------------------------------------- #

EPSILON = 1e-12


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float,
    max_iter: int,
) -> MethodResult:
    result = _base_result(["iteration", "a", "b", "c", "f(c)", "error"])
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return _finalize(
            result,
            converged=False,
            root=None,
            error=None,
            message="f(a) and f(b) must have opposite signs for bisection.",
        )

    prev_c = None
    for i in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = f(c)
        error = abs(c - prev_c) if prev_c is not None else None
        result.iterations.append(
            {
                "iteration": i,
                "a": a,
                "b": b,
                "c": c,
                "f(c)": fc,
                "error": error if error is not None else float("nan"),
            }
        )
        if abs(fc) <= tol or (error is not None and error <= tol):
            return _finalize(
                result,
                converged=True,
                root=c,
                error=abs(fc),
                message=f"Converged in {i} iterations.",
            )
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        prev_c = c

    return _finalize(
        result,
        converged=False,
        root=c,
        error=abs(fc),
        message="Maximum iterations reached without convergence.",
    )


def regula_falsi(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float,
    max_iter: int,
) -> MethodResult:
    result = _base_result(["iteration", "a", "b", "c", "f(c)", "error"])
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return _finalize(
            result,
            converged=False,
            root=None,
            error=None,
            message="f(a) and f(b) must have opposite signs for Regula Falsi.",
        )

    c = None
    fc = None
    for i in range(1, max_iter + 1):
        denominator = fb - fa
        if abs(denominator) < EPSILON:
            return _finalize(
                result,
                converged=False,
                root=None,
                error=None,
                message="Division by zero encountered in Regula Falsi.",
            )
        c = b - fb * (b - a) / denominator
        fc = f(c)
        error = abs(fc)
        result.iterations.append(
            {"iteration": i, "a": a, "b": b, "c": c, "f(c)": fc, "error": error}
        )
        if abs(fc) <= tol:
            return _finalize(
                result,
                converged=True,
                root=c,
                error=error,
                message=f"Converged in {i} iterations.",
            )
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    return _finalize(
        result,
        converged=False,
        root=c,
        error=abs(fc) if fc is not None else None,
        message="Maximum iterations reached without convergence.",
    )


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float,
    max_iter: int,
) -> MethodResult:
    result = _base_result(
        ["iteration", "x_prev", "x_curr", "x_next", "f(x_curr)", "error"]
    )
    fx0, fx1 = f(x0), f(x1)
    for i in range(1, max_iter + 1):
        denominator = (fx1 - fx0)
        if abs(denominator) < EPSILON:
            return _finalize(
                result,
                converged=False,
                root=x1,
                error=abs(fx1),
                message="Division by zero encountered in Secant method.",
            )
        x2 = x1 - fx1 * (x1 - x0) / denominator
        fx2 = f(x2)
        error = abs(x2 - x1)
        result.iterations.append(
            {
                "iteration": i,
                "x_prev": x0,
                "x_curr": x1,
                "x_next": x2,
                "f(x_curr)": fx1,
                "error": error,
            }
        )
        if abs(fx2) <= tol or error <= tol:
            return _finalize(
                result,
                converged=True,
                root=x2,
                error=abs(fx2),
                message=f"Converged in {i} iterations.",
            )
        x0, fx0 = x1, fx1
        x1, fx1 = x2, fx2

    return _finalize(
        result,
        converged=False,
        root=x2,
        error=abs(fx2),
        message="Maximum iterations reached without convergence.",
    )


def newton_raphson(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float,
    max_iter: int,
) -> MethodResult:
    result = _base_result(["iteration", "x", "f(x)", "f'(x)", "error"])
    x = x0
    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < EPSILON:
            return _finalize(
                result,
                converged=False,
                root=x,
                error=abs(fx),
                message="Derivative too small; Newton-Raphson cannot proceed.",
            )
        x_new = x - fx / dfx
        error = abs(x_new - x)
        result.iterations.append(
            {"iteration": i, "x": x, "f(x)": fx, "f'(x)": dfx, "error": error}
        )
        if abs(fx) <= tol or error <= tol:
            return _finalize(
                result,
                converged=True,
                root=x_new,
                error=abs(fx),
                message=f"Converged in {i} iterations.",
            )
        x = x_new

    return _finalize(
        result,
        converged=False,
        root=x,
        error=abs(f(x)),
        message="Maximum iterations reached without convergence.",
    )


def fixed_point_iteration(
    g: Callable[[float], float],
    x0: float,
    tol: float,
    max_iter: int,
) -> MethodResult:
    result = _base_result(["iteration", "x", "g(x)", "error"])
    x = x0
    for i in range(1, max_iter + 1):
        gx = g(x)
        error = abs(gx - x)
        result.iterations.append({"iteration": i, "x": x, "g(x)": gx, "error": error})
        if error <= tol:
            return _finalize(
                result,
                converged=True,
                root=gx,
                error=error,
                message=f"Converged in {i} iterations.",
            )
        x = gx

    return _finalize(
        result,
        converged=False,
        root=x,
        error=error,
        message="Maximum iterations reached without convergence.",
    )


def modified_secant(
    f: Callable[[float], float],
    x0: float,
    delta: float,
    tol: float,
    max_iter: int,
) -> MethodResult:
    result = _base_result(["iteration", "x", "f(x)", "x_next", "error"])
    x = x0
    for i in range(1, max_iter + 1):
        fx = f(x)
        perturbation = delta * x if x != 0 else delta
        denominator = f(x + perturbation) - fx
        if abs(denominator) < EPSILON:
            return _finalize(
                result,
                converged=False,
                root=x,
                error=abs(fx),
                message="Division by zero in Modified Secant method.",
            )
        x_new = x - fx * perturbation / denominator
        error = abs(x_new - x)
        result.iterations.append(
            {"iteration": i, "x": x, "f(x)": fx, "x_next": x_new, "error": error}
        )
        if abs(fx) <= tol or error <= tol:
            return _finalize(
                result,
                converged=True,
                root=x_new,
                error=abs(fx),
                message=f"Converged in {i} iterations.",
            )
        x = x_new

    return _finalize(
        result,
        converged=False,
        root=x,
        error=abs(f(x)),
        message="Maximum iterations reached without convergence.",
    )


# --------------------------------------------------------------------------- #
# Public runner
# --------------------------------------------------------------------------- #

MethodParams = Dict[str, float]


def run_method(
    method: str,
    *,
    function_expr: str,
    params: MethodParams,
    g_expr: Optional[str] = None,
) -> MethodResult:
    """
    Dispatch helper that evaluates the chosen numerical method.

    Parameters
    ----------
    method : Literal key identifying the algorithm.
    function_expr : f(x) expression supplied by the user.
    params : numeric values needed by the specific method.
    g_expr : optional g(x) expression for Fixed Point iteration.
    """

    method = method.lower()
    f = build_function(function_expr)

    if method == "bisection":
        return bisection(
            f,
            a=params["lower"],
            b=params["upper"],
            tol=params["tolerance"],
            max_iter=int(params["max_iterations"]),
        )
    if method == "regula_falsi":
        return regula_falsi(
            f,
            a=params["lower"],
            b=params["upper"],
            tol=params["tolerance"],
            max_iter=int(params["max_iterations"]),
        )
    if method == "secant":
        return secant(
            f,
            x0=params["x0"],
            x1=params["x1"],
            tol=params["tolerance"],
            max_iter=int(params["max_iterations"]),
        )
    if method == "newton_raphson":
        df = build_derivative(function_expr)
        return newton_raphson(
            f,
            df,
            x0=params["initial_guess"],
            tol=params["tolerance"],
            max_iter=int(params["max_iterations"]),
        )
    if method == "fixed_point":
        if not g_expr:
            raise ExpressionParseError("g(x) expression is required for Fixed Point.")
        g = build_function(g_expr)
        return fixed_point_iteration(
            g,
            x0=params["initial_guess"],
            tol=params["tolerance"],
            max_iter=int(params["max_iterations"]),
        )
    if method == "modified_secant":
        return modified_secant(
            f,
            x0=params["initial_guess"],
            delta=params["delta"],
            tol=params["tolerance"],
            max_iter=int(params["max_iterations"]),
        )

    raise ValueError(f"Unknown method: {method}")


# Mapping useful for UI layers
METHOD_LABELS = {
    "bisection": "Bisection Method",
    "regula_falsi": "Regula Falsi (False Position)",
    "secant": "Secant Method",
    "newton_raphson": "Newton–Raphson Method",
    "fixed_point": "Fixed Point Iteration",
    "modified_secant": "Modified Secant Method",
}


