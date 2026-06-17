from pathlib import Path

from hypnodata.config import load_config


def test_hypnodata_toy_config_and_readme_document_boundaries():
    root = Path(__file__).resolve().parents[1]

    config = load_config(root / "configs" / "hypnodata" / "toy_edf_npz.yaml")
    readme = (root / "configs" / "hypnodata" / "README.md").read_text()

    assert config.center == "toy"
    assert config.backend.type == "npz"
    assert "sleep2stat.io.records.load_records" in readme
    assert "convert_npz_to_kaldi.py" in readme
    assert "does not write ark/scp" in readme
    assert "does not compute" in readme
