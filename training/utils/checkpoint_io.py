"""Fail-closed checkpoint loading shared by training, evaluation and rendering.

Policy (beta-test item B2):
  * a checkpoint that was asked for but is absent or unreadable stops the run
    (``FileNotFoundError`` / ``RuntimeError``) — never a silent fall-back to
    randomly initialised weights;
  * a checkpoint whose tensors do not match the model stops the run with a
    message that lists the missing, unexpected and shape-mismatched keys —
    never a ``strict=False`` retry;
  * partial loading is allowed only through an explicit allowlist of
    parameter-name prefixes (``allow_prefixes``), and *every* key inside that
    allowlist must be present with the right shape.  Keys outside the
    allowlist are left untouched in the model / skipped from the checkpoint
    and are reported, so an intentional partial load is visible in the log.

This module is deliberately pure torch with no project imports:
``rendering/brdf_plugin/mlp.py`` loads it by file path (the rendering package
does not import ``training``), so both trees share one implementation.
"""
import os
from collections import OrderedDict

import torch

# Keys listed per group in an error message before "... and N more".
MAX_LISTED_KEYS = 20


class IncompatibleCheckpointError(RuntimeError):
    """Checkpoint tensors do not match the model (missing / unexpected / shape)."""


def load_checkpoint_file(path, map_location='cpu'):
    """``torch.load`` that fails closed.

    Raises ``FileNotFoundError`` (naming the path) when the file is missing
    and ``RuntimeError`` when it cannot be unpickled.  Loads with
    ``weights_only=False`` because Lightning checkpoints embed the composed
    Hydra config under ``hyper_parameters``.
    """
    if path is None or str(path) == "":
        raise FileNotFoundError("no checkpoint path was given")
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as e:  # unpickling / corrupt zip / wrong file type
        raise RuntimeError(f"failed to read checkpoint {path}: {e}") from e


def extract_state_dict(ckpt, source="checkpoint"):
    """Return the tensor state dict inside a loaded checkpoint object.

    Accepts a Lightning checkpoint (``{'state_dict': ...}``), the plain
    ``{'model_state_dict': ...}`` convention, or a bare state dict.  Anything
    else (or non-tensor values) is an error rather than "no weights".
    """
    if not isinstance(ckpt, dict):
        raise RuntimeError(
            f"{source}: unsupported checkpoint object of type "
            f"{type(ckpt).__name__}; expected a dict / state dict")
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"{source}: 'state_dict' entry is a {type(state_dict).__name__}, not a dict")
    non_tensor = [k for k, v in state_dict.items() if not torch.is_tensor(v)]
    if non_tensor:
        raise RuntimeError(
            f"{source}: state dict has {len(non_tensor)} non-tensor value(s): "
            f"{_truncate(non_tensor)}")
    return state_dict


def split_namespace(state_dict, prefix):
    """Split ``state_dict`` into ``(inside, outside)`` by key prefix.

    ``inside`` holds the keys that start with ``prefix`` (prefix stripped,
    original order kept); ``outside`` holds every other key unchanged.
    """
    inside, outside = OrderedDict(), OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            inside[k[len(prefix):]] = v
        else:
            outside[k] = v
    return inside, outside


class LoadReport:
    """What ``load_state_dict_strict`` did: loaded / skipped / untouched keys."""

    def __init__(self, source, loaded, skipped, untouched, allow_prefixes, ignore_prefixes):
        self.source = source
        self.loaded = list(loaded)          # model keys overwritten from the checkpoint
        self.skipped = list(skipped)        # checkpoint keys deliberately not loaded
        self.untouched = list(untouched)    # model keys deliberately kept as they were
        self.allow_prefixes = tuple(allow_prefixes) if allow_prefixes else None
        self.ignore_prefixes = tuple(ignore_prefixes)

    def summary(self, max_items=6):
        scope = (", ".join(p + "*" for p in self.allow_prefixes)
                 if self.allow_prefixes else "all model keys")
        lines = [f"=> loaded {len(self.loaded)} tensors from {self.source} (scope: {scope})"]
        if self.skipped:
            lines.append(f"   skipped {len(self.skipped)} checkpoint keys outside the scope: "
                         f"{_truncate(self.skipped, max_items)}")
        if self.untouched:
            lines.append(f"   kept {len(self.untouched)} model keys at their current values: "
                         f"{_truncate(self.untouched, max_items)}")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()


def load_state_dict_strict(model, state_dict, *, allow_prefixes=None, ignore_prefixes=(),
                           source="checkpoint"):
    """Copy ``state_dict`` into ``model`` or raise — never a partial load.

    Args:
        model: target ``torch.nn.Module``.
        state_dict: mapping of parameter name -> tensor (already unwrapped and
            prefix-cleaned by the caller).
        allow_prefixes: ``None`` = full strict load (checkpoint must contain
            exactly the model's keys).  A tuple of key prefixes = *intentional
            partial load*: the scope is every model key that starts with one of
            them, all scoped keys must be present with matching shapes,
            checkpoint keys inside the scope that the model lacks are errors,
            and everything outside the scope is left alone (model) / skipped
            (checkpoint) and reported.  E.g. ``('material.decoder.',)`` is the
            stage-2 "decoder-only warm start".
        ignore_prefixes: explicit namespaces excluded from the comparison on
            both sides, for state that the model rebuilds itself (e.g.
            ``('emitter.',)``: light geometry rebuilt from calibration files).
        source: label for messages (normally the checkpoint path).

    Returns:
        ``LoadReport`` describing what was loaded, skipped and left untouched.

    Raises:
        IncompatibleCheckpointError (a ``RuntimeError``) listing missing,
        unexpected and shape-mismatched keys, grouped and truncated.
    """
    if allow_prefixes is not None:
        allow_prefixes = tuple(allow_prefixes)
        if len(allow_prefixes) == 0:
            raise ValueError("allow_prefixes must be None (strict) or a non-empty tuple")
    ignore_prefixes = tuple(ignore_prefixes)

    def in_scope(key):
        if ignore_prefixes and key.startswith(ignore_prefixes):
            return False
        return allow_prefixes is None or key.startswith(allow_prefixes)

    model_sd = model.state_dict()
    scoped_model = [k for k in model_sd if in_scope(k)]
    scoped_ckpt = [k for k in state_dict if in_scope(k)]
    if allow_prefixes is not None and not scoped_model:
        raise IncompatibleCheckpointError(
            f"cannot load {source} into {type(model).__name__}: the allowlist "
            f"{allow_prefixes} matches no model parameter (model keys start with "
            f"{_truncate(sorted({k.split('.')[0] + '.' for k in model_sd}), 8)})")

    missing = [k for k in scoped_model if k not in state_dict]
    unexpected = [k for k in scoped_ckpt if k not in model_sd]
    mismatched = [
        (k, tuple(state_dict[k].shape), tuple(model_sd[k].shape))
        for k in scoped_model
        if k in state_dict and tuple(state_dict[k].shape) != tuple(model_sd[k].shape)
    ]
    if missing or unexpected or mismatched:
        raise IncompatibleCheckpointError(_format_incompatible(
            source, type(model).__name__, allow_prefixes, ignore_prefixes,
            missing, unexpected, mismatched))

    # Same mechanics as the original hand-written loaders (model dict updated
    # with the selected tensors, then a strict load) so a successful load is
    # numerically identical to before.
    merged = OrderedDict(model_sd)
    for k in scoped_model:
        merged[k] = state_dict[k]
    model.load_state_dict(merged, strict=True)

    scoped_ckpt_set = set(scoped_ckpt)
    scoped_model_set = set(scoped_model)
    return LoadReport(
        source=source,
        loaded=scoped_model,
        skipped=[k for k in state_dict if k not in scoped_ckpt_set],
        untouched=[k for k in model_sd if k not in scoped_model_set],
        allow_prefixes=allow_prefixes,
        ignore_prefixes=ignore_prefixes,
    )


# --------------------------------------------------------------------------
# message formatting
# --------------------------------------------------------------------------
def _truncate(items, limit=MAX_LISTED_KEYS):
    items = list(items)
    shown = ", ".join(str(i) for i in items[:limit])
    if len(items) > limit:
        shown += f", ... and {len(items) - limit} more"
    return shown


def _format_group(title, entries):
    lines = [f"  {title} ({len(entries)}):"]
    for e in entries[:MAX_LISTED_KEYS]:
        lines.append(f"    {e}")
    if len(entries) > MAX_LISTED_KEYS:
        lines.append(f"    ... and {len(entries) - MAX_LISTED_KEYS} more")
    return lines


def _format_incompatible(source, model_name, allow_prefixes, ignore_prefixes,
                         missing, unexpected, mismatched):
    scope = (", ".join(p + "*" for p in allow_prefixes) if allow_prefixes else "all model keys")
    head = f"cannot load {source} into {model_name} (scope: {scope}"
    if ignore_prefixes:
        head += "; ignored: " + ", ".join(p + "*" for p in ignore_prefixes)
    head += "):"
    lines = [head]
    if missing:
        lines += _format_group("missing from checkpoint", missing)
    if unexpected:
        lines += _format_group("unexpected in checkpoint", unexpected)
    if mismatched:
        lines += _format_group("shape mismatch", [
            f"{k}: checkpoint {tuple(ck)} vs model {tuple(md)}" for k, ck, md in mismatched])
    lines.append("Refusing to load an incompatible checkpoint (no strict=False fallback). "
                 "Check that the checkpoint was produced by this architecture/config.")
    return "\n".join(lines)
