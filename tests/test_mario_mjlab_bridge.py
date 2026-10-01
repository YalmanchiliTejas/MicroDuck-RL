import ast
from pathlib import Path

from mjlab_microduck.super_mario_bridge import FlybrainRequest


ROOT = Path(__file__).parents[1]


def _load_request_name():
    path = ROOT / "scripts/run_mario_mjlab_bridge.py"
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "request_name"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["request_name"]


def test_exact_bridge_reports_all_flybrain_intents():
    request_name = _load_request_name()
    assert request_name(FlybrainRequest()) == "idle"
    assert request_name(FlybrainRequest(left=True)) == "left"
    assert request_name(FlybrainRequest(right=True, run=True)) == "right_run"
    assert request_name(FlybrainRequest(jump=True)) == "jump"
    assert (
        request_name(FlybrainRequest(left=True, jump=True, run=True))
        == "left_run_jump"
    )


def test_combined_launcher_uses_exact_mjlab_bridge():
    source = (ROOT / "scripts/run_mario_flybrain.py").read_text()
    assert '"scripts/run_mario_mjlab_bridge.py"' in source
    assert '"scripts/infer_policy.py"' not in source
    assert '"watching rollouts in "' in source
