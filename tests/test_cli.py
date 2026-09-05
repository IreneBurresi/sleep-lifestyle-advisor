import json
from pathlib import Path

from advisor.cli import build_testset, evaluate, generate_guidance


def test_generate_guidance_show_prompt(artifacts, capsys):
    assert generate_guidance.main(["--cluster", "3", "--show-prompt"]) == 0
    out = capsys.readouterr().out
    assert "## Group profile (cluster 3" in out and "## Fixed flags" in out


def test_generate_guidance_rejects_bad_user(tmp_path: Path, artifacts, capsys):
    bad = tmp_path / "u.json"
    bad.write_text(json.dumps({"age": 12}))
    assert generate_guidance.main(["--user", str(bad)]) == 1
    assert "age" in capsys.readouterr().err


def test_build_testset_writes_nine_cases(tmp_path: Path, artifacts, raw):
    out = tmp_path / "t.json"
    assert build_testset.main(["--out", str(out)]) == 0
    cases = json.loads(out.read_text())
    assert len(cases) == 9 and len({c["row"] for c in cases}) == 9


def test_evaluate_checks_offline(tmp_path: Path, artifacts):
    assert evaluate.main(["eval/runs/flash-lite_v2.json", "--out", str(tmp_path)]) == 0
    rows = json.loads(next(tmp_path.glob("checks_*.json")).read_text())
    assert len(rows) == 9 and all("no medical content" in r for r in rows)
