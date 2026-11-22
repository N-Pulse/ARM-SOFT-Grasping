import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "colab" / "colab_train.ipynb"


def _source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_colab_notebook_defaults_to_versioned_mixed_continuation():
    source = _source()
    assert 'DATASET_NAME = "openimages6_d405_v1"' in source
    assert 'START_MODEL_DRIVE = "models/yolo26s-mix-openimages-d405-v1.0.0/best.pt"' in source
    assert 'OPTIMIZER = "AdamW"' in source
    assert "LEARNING_RATE = 0.0002" in source


def test_colab_notebook_guards_gpu_inputs_classes_and_run_name():
    source = _source()
    assert "torch.cuda.is_available()" in source
    assert "Upload the missing files to Drive" in source
    assert "Unexpected model classes" in source
    assert "Run already exists" in source
    assert "subprocess.run(command, cwd=REPO, check=True)" in source
