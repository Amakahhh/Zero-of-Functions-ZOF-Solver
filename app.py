"""
Flask web GUI for the Zero of Functions (ZOF) Solver.

Users can:
    - Pick any of the six numerical methods.
    - Enter f(x) (and g(x) where required) plus method-specific parameters.
    - Review per-iteration diagnostics and the final estimated root.
"""

from __future__ import annotations

from typing import Dict, Optional

from flask import Flask, render_template, request

from zof_solver import (
    ExpressionParseError,
    METHOD_LABELS,
    MethodResult,
    run_method,
)

app = Flask(__name__)

DEFAULTS = {
    "function_expr": "x**3 - x - 2",
    "g_expr": "cos(x)",
    "lower": "-2",
    "upper": "2",
    "x0": "0",
    "x1": "1",
    "initial_guess": "1",
    "delta": "0.01",
    "tolerance": "1e-6",
    "max_iterations": "50",
}


def _float_from_form(name: str, form, default: Optional[float] = None) -> float:
    raw = form.get(name)
    if raw is None or raw.strip() == "":
        if default is None:
            raise ValueError(f"{name.replace('_', ' ').title()} is required.")
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name.replace('_', ' ').title()} must be numeric.") from exc


def _int_from_form(name: str, form, default: Optional[int] = None) -> int:
    raw = form.get(name)
    if raw is None or raw.strip() == "":
        if default is None:
            raise ValueError(f"{name.replace('_', ' ').title()} is required.")
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name.replace('_', ' ').title()} must be an integer.") from exc


def _collect_params(method: str, form) -> Dict[str, float]:
    params: Dict[str, float] = {
        "tolerance": _float_from_form("tolerance", form, default=1e-6),
        "max_iterations": _int_from_form("max_iterations", form, default=50),
    }

    if method in {"bisection", "regula_falsi"}:
        params["lower"] = _float_from_form("lower", form)
        params["upper"] = _float_from_form("upper", form)
    elif method == "secant":
        params["x0"] = _float_from_form("x0", form)
        params["x1"] = _float_from_form("x1", form)
    elif method in {"newton_raphson", "fixed_point", "modified_secant"}:
        params["initial_guess"] = _float_from_form("initial_guess", form)
        if method == "modified_secant":
            params["delta"] = _float_from_form("delta", form, default=1e-4)
    else:
        raise ValueError(f"Unsupported method: {method}")

    if method == "fixed_point":
        g_expr = form.get("g_expr", "").strip()
        if not g_expr:
            raise ValueError("g(x) is required for Fixed Point Iteration.")
        params["g_expr"] = g_expr

    return params


@app.route("/", methods=["GET", "POST"])
def index():
    result: Optional[MethodResult] = None
    error_message: Optional[str] = None
    selected_method = request.form.get("method", "bisection")
    function_expr = request.form.get("function_expr", DEFAULTS["function_expr"])
    g_expr_value = request.form.get("g_expr", DEFAULTS["g_expr"])

    if request.method == "POST":
        function_expr = request.form.get("function_expr", "").strip()
        if not function_expr:
            error_message = "Please provide f(x)."
        else:
            try:
                params = _collect_params(selected_method, request.form)
                g_expr = params.pop("g_expr", None)
                result = run_method(
                    selected_method,
                    function_expr=function_expr,
                    params=params,  # type: ignore[arg-type]
                    g_expr=g_expr or g_expr_value,
                )
            except (ExpressionParseError, ValueError) as exc:
                error_message = str(exc)

    context = {
        "methods": METHOD_LABELS,
        "selected_method": selected_method,
        "function_expr": function_expr,
        "g_expr": g_expr_value,
        "defaults": DEFAULTS,
        "result": result,
        "error_message": error_message,
    }
    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True)


