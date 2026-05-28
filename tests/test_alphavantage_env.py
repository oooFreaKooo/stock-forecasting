import os
from pathlib import Path

from radar.data.adapters import alphavantage as av


def test_repo_root_is_project_not_src():
    root = av._REPO_ROOT
    assert (root / "pyproject.toml").is_file()
    assert not (root / "radar").is_dir()


def test_load_project_env_reads_alphavantage_key(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("ALPHAVANTAGE_API_KEY=test-key-from-file\n", encoding="utf-8")
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    av._ENV_LOADED = False
    monkeypatch.setattr(av, "_REPO_ROOT", tmp_path)
    av._load_project_env()
    assert os.environ.get("ALPHAVANTAGE_API_KEY") == "test-key-from-file"
    assert av.is_configured()


def test_load_project_env_uses_radar_root_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("ALPHAVANTAGE_API_KEY=from-radar-root\n", encoding="utf-8")
    monkeypatch.setenv("RADAR_ROOT", str(tmp_path))
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    av._ENV_LOADED = False
    av._load_project_env()
    assert os.environ.get("ALPHAVANTAGE_API_KEY") == "from-radar-root"
