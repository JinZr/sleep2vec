from __future__ import annotations

from agent_tools import repo


def test_repo_summary_uses_shared_index_on_feature_branch(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / "doc/codex_index").mkdir(parents=True)
    responses = {
        ("branch", "--show-current"): (True, "feature/lightweight-index"),
        ("rev-parse", "HEAD"): (True, "abc123"),
        ("status", "--short"): (True, ""),
    }
    monkeypatch.setattr(repo, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(repo, "_git", lambda args: responses[tuple(args)])

    summary = repo.repo_summary()

    assert summary["git"]["branch"] == "feature/lightweight-index"
    assert summary["codex_index"] == {"path": "doc/codex_index", "exists": True}
    assert "branch_index_path" not in summary["codex_index"]
    assert "fallback_main_exists" not in summary["codex_index"]
