"""
Command-Line Interface for the Zero of Functions (ZOF) Solver.

The CLI walks beginners through:
    1. Choosing any of the six supported numerical methods.
    2. Entering the required function(s) and initial parameters.
    3. Viewing per-iteration diagnostics plus the final estimated root.

The same numerical core is shared with the Flask web interface to keep the
project maintainable.
"""

from __future__ import annotations

import math
from typing import Dict, List

from zof_solver import (
    ExpressionParseError,
    METHOD_LABELS,
    MethodResult,
    run_method,
)


def _format_value(value) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        if math.isnan(value):
            return "--"
    try:
        number = float(value)
        return f"{number:.6g}"
    except (TypeError, ValueError):
        return str(value)


def _print_iterations(result: MethodResult) -> None:
    if not result.iterations:
        print("No iteration details to display.")
        return
    columns = result.columns
    rows: List[List[str]] = []
    for entry in result.iterations:
        rows.append([_format_value(entry.get(col)) for col in columns])

    widths = [
        max(len(col), *(len(row[idx]) for row in rows))
        for idx, col in enumerate(columns)
    ]

    def print_row(values: List[str]) -> None:
        line = " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))
        print(line)

    print("\nDetailed Iterations")
    print("-" * (sum(widths) + 3 * (len(columns) - 1)))
    print_row(columns)
    print("-" * (sum(widths) + 3 * (len(columns) - 1)))
    for row in rows:
        print_row(row)
    print("-" * (sum(widths) + 3 * (len(columns) - 1)))


def _prompt_float(message: str, *, default: float | None = None) -> float:
    while True:
        raw = input(f"{message} " + (f"[default: {default}] " if default else ""))
        if not raw.strip():
            if default is not None:
                return default
            print("Value is required. Please try again.")
            continue
        try:
            return float(raw)
        except ValueError:
            print("Invalid number. Please enter a numeric value.")


def _prompt_int(message: str, *, default: int | None = None) -> int:
    while True:
        raw = input(f"{message} " + (f"[default: {default}] " if default else ""))
        if not raw.strip():
            if default is not None:
                return default
            print("Value is required. Please try again.")
            continue
        try:
            return int(raw)
        except ValueError:
            print("Invalid integer. Please enter a whole number.")


def _prompt_function(prompt: str) -> str:
    while True:
        expr = input(prompt).strip()
        if expr:
            return expr
        print("Expression cannot be empty. Please try again.")


def _collect_method_params(method_key: str) -> Dict[str, float | int | str]:
    common_tol = _prompt_float("Enter tolerance (e.g., 1e-6):", default=1e-6)
    max_iter = _prompt_int("Enter maximum iterations:", default=50)

    params: Dict[str, float | int | str] = {
        "tolerance": common_tol,
        "max_iterations": max_iter,
    }

    if method_key in {"bisection", "regula_falsi"}:
        params["lower"] = _prompt_float("Enter lower bound (a):")
        params["upper"] = _prompt_float("Enter upper bound (b):")
    elif method_key == "secant":
        params["x0"] = _prompt_float("Enter first initial guess (x0):")
        params["x1"] = _prompt_float("Enter second initial guess (x1):")
    elif method_key in {"newton_raphson", "fixed_point", "modified_secant"}:
        params["initial_guess"] = _prompt_float("Enter initial guess:")
        if method_key == "modified_secant":
            params["delta"] = _prompt_float("Enter perturbation (delta):", default=1e-4)
    else:
        raise ValueError(f"Unsupported method {method_key}")

    if method_key == "fixed_point":
        params["g_expr"] = _prompt_function(
            "Enter g(x) for Fixed Point (e.g., cos(x)):"
        )

    return params


def _display_summary(result: MethodResult) -> None:
    print("\nSummary")
    print("-------")
    print(f"Status       : {'Converged' if result.converged else 'Did not converge'}")
    print(f"Estimated root: {_format_value(result.root)}")
    print(f"Final error  : {_format_value(result.final_error)}")
    print(f"Iterations   : {len(result.iterations)}")
    print(f"Message      : {result.message}")


def main() -> None:
    print("=" * 70)
    print("Zero of Functions (ZOF) Solver - CLI")
    print("Enter equations using the variable x. Example: x**3 - 5*x + 2 or sin(x)")
    print("=" * 70)

    method_keys = list(METHOD_LABELS.keys())

    while True:
        print("\nAvailable Methods:")
        for idx, key in enumerate(method_keys, start=1):
            print(f"  {idx}. {METHOD_LABELS[key]}")
        print("  0. Exit")
        shortcut_hint = ", ".join(
            f"{idx}={METHOD_LABELS[key].split(' (')[0]}"
            for idx, key in enumerate(method_keys, start=1)
        )
        print(f"\nHint: choose a number (e.g., {shortcut_hint}).")

        choice_raw = input("\nSelect a method by number: ").strip()
        if choice_raw == "0":
            print("Goodbye!")
            break
        try:
            choice = int(choice_raw)
            method_key = method_keys[choice - 1]
        except (ValueError, IndexError):
            print("Invalid selection. Please choose a valid method number.")
            continue

        function_expr = _prompt_function("Enter f(x): ")

        try:
            params = _collect_method_params(method_key)
            g_expr = params.pop("g_expr", None)
            result = run_method(
                method_key,
                function_expr=function_expr,
                params=params,  # type: ignore[arg-type]
                g_expr=g_expr,
            )
        except ExpressionParseError as exc:
            print(f"Input error: {exc}")
            continue
        except Exception as exc:  # pragma: no cover - general safeguard
            print(f"Something went wrong: {exc}")
            continue

        _print_iterations(result)
        _display_summary(result)

        again = input("\nWould you like to solve another equation? (y/n): ").strip()
        if again.lower() not in {"y", "yes"}:
            print("Thanks for using the ZOF Solver!")
            break


if __name__ == "__main__":
    main()


