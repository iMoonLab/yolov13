# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path

from . import USER_CONFIG_DIR
from .torch_utils import TORCH_1_9

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_free_network_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serialize_overrides(overrides: dict) -> dict:
    """Serialize trainer overrides for DDP subprocess compatibility."""
    serialized = overrides.copy()
    augmentations = serialized.get("augmentations")
    if augmentations is not None:
        try:
            import albumentations as A

            serialized["augmentations"] = [A.to_dict(t) for t in augmentations]
            serialized["_augmentations_serialized"] = True
        except Exception:
            serialized["augmentations"] = None
            serialized["_augmentations_serialized"] = False
    return serialized


def generate_ddp_file(trainer):
    """Generate temporary python entrypoint for DDP subprocess."""
    module, name = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)
    overrides = _serialize_overrides(vars(trainer.args))

    content = f"""
# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)
import sys
from pathlib import Path, PosixPath, WindowsPath

repo_root = Path(r"{PROJECT_ROOT}")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import os

overrides = {overrides}

if "use_turing_flash" in overrides:
    os.environ["Y13_USE_TURING_FLASH"] = "1" if overrides["use_turing_flash"] else "0"
if "force_disable_flash" in overrides:
    os.environ["Y13_DISABLE_FLASH"] = "1" if overrides["force_disable_flash"] else "0"

if __name__ == "__main__":
    from {module} import {name}
    from ultralytics.utils import DEFAULT_CFG_DICT

    if overrides.pop("_augmentations_serialized", False) and overrides.get("augmentations") is not None:
        import albumentations as A

        overrides["augmentations"] = [A.from_dict(t) for t in overrides["augmentations"]]

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')
    trainer = {name}(cfg=cfg, overrides=overrides)
    trainer.args.model = "{getattr(trainer.hub_session, "model_url", trainer.args.model)}"
    trainer.train()
"""

    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(trainer)}.py",
        mode="w+",
        encoding="utf-8",
        dir=USER_CONFIG_DIR / "DDP",
        delete=False,
    ) as file:
        file.write(content)
    return file.name


def generate_ddp_command(world_size, trainer):
    """Generate command tuple for distributed training."""
    import __main__  # noqa: F401

    if not trainer.resume and trainer.save_dir.exists():
        shutil.rmtree(trainer.save_dir, ignore_errors=True)
    file = generate_ddp_file(trainer)
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    port = find_free_network_port()
    cmd = [sys.executable, "-m", dist_cmd, "--nproc_per_node", f"{world_size}", "--master_port", f"{port}", file]
    return cmd, file


def ddp_cleanup(trainer, file):
    """Delete temp DDP file if created."""
    if f"{id(trainer)}.py" in file and os.path.exists(file):
        os.remove(file)
