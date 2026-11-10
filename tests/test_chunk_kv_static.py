"""Regression guard for tail chunks; this test imports no CacheGen dependencies."""
import ast
from pathlib import Path


def test_chunk_kv_iterates_over_tail_and_bounds_slice():
    source = (Path(__file__).parents[1] / "src" / "cachegen_interface.py").read_text()
    tree = ast.parse(source)
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "chunk_kv"
    )
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    assert any(
        isinstance(call.func, ast.Name)
        and call.func.id == "range"
        and len(call.args) == 3
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == 0
        and isinstance(call.args[1], ast.Name)
        and call.args[1].id == "ntokens"
        and isinstance(call.args[2], ast.Name)
        and call.args[2].id == "chunk_size"
        for call in calls
    ), "chunk_kv must iterate range(0, ntokens, chunk_size), including a tail"
    assert any(
        isinstance(node, ast.Slice)
        and isinstance(node.upper, ast.Call)
        and isinstance(node.upper.func, ast.Name)
        and node.upper.func.id == "min"
        for node in ast.walk(function)
    ), "the final chunk slice must be bounded by ntokens"
