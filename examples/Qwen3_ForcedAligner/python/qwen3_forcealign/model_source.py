"""Resolve local checkpoints or download model IDs for all export steps."""

from pathlib import Path


def resolve_model_path(model_path, modelscope=False):
    """Prefer a local directory; remote downloads use the provider cache."""
    local = Path(model_path).expanduser()
    if local.is_dir():
        resolved = local.resolve()
    elif local.exists():
        raise NotADirectoryError(f"Model path is not a directory: {local}")
    elif local.is_absolute() or str(model_path).startswith((".", "~")):
        raise FileNotFoundError(f"Model directory does not exist: {local}")
    else:
        if modelscope:
            from modelscope import snapshot_download

            resolved = Path(snapshot_download(str(model_path))).resolve()
        else:
            from huggingface_hub import snapshot_download

            resolved = Path(
                snapshot_download(repo_id=str(model_path))
            ).resolve()
    if not (resolved / "config.json").is_file():
        raise FileNotFoundError(f"Missing model config: {resolved}")
    return resolved
