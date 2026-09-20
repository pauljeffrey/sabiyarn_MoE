"""MLflow experiment tracking for training runs (Modal, vast.ai, anywhere).

Kept deliberately torch-free and import-safe: `mlflow` is imported lazily
inside `MlflowTracker.start`, and every call swallows/logs its own errors, so
a missing package, an unreachable tracking server or a bad URI can never take
down a training run -- tracking just turns itself off with a warning.

Where runs are stored (first match wins):
  1. `MLFLOW_TRACKING_URI` env var (e.g. `http://my-server:5000`, or
     `file:/data/mlruns` -- this is how Modal points it at the volume)
  2. `mlflow.tracking_uri` in train_config.yaml
  3. `./mlruns` relative to the working directory (the vast.ai default)

View with:  MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri <same uri> --port 5000
"""

from __future__ import annotations

import atexit
import math
import os
import socket
import subprocess
import sys
from typing import Any, Iterable, Optional

import structlog

LOG = structlog.get_logger()

LN2 = math.log(2.0)

# MLflow truncates/rejects param values past a few hundred chars on older servers.
_MAX_PARAM_LEN = 500


def build_token_byte_lengths(
    vocab: dict[str, int],
    special_ids: Iterable[int],
    table_size: Optional[int] = None,
) -> list[int]:
    """Number of UTF-8 bytes each token id contributes to decoded text.

    The tokenizer is byte-level BPE (GPT-2 style byte<->unicode mapping), where
    every character of a token string stands for exactly one raw byte, so a
    regular token's byte length is just `len(token_string)`. Special/added
    tokens (`<|user|>`, `</s>`, `<tool_call>`, ...) are structural, not text,
    and count as 0 bytes -- the same convention as nanochat's bpb metric.

    Returns a list of length `table_size + 1` (default: max id + 1, plus one
    trailing 0 slot). Ids >= table_size should be clamped to the last index by
    the caller so out-of-vocab/padded ids resolve to 0 instead of erroring.
    """
    special = set(special_ids)
    n = table_size if table_size is not None else (max(vocab.values()) + 1 if vocab else 0)
    table = [0] * (n + 1)
    for token, idx in vocab.items():
        if idx < n and idx not in special:
            table[idx] = len(token)
    return table


def bits_per_byte(mean_ce_nats: float, n_tokens: float, n_bytes: float) -> Optional[float]:
    """Tokenizer-independent loss: total nats over the supervised tokens,
    converted to bits, divided by the raw bytes those tokens decode to."""
    if n_bytes <= 0:
        return None
    return mean_ce_nats * n_tokens / (LN2 * n_bytes)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        else:
            out[key] = str(v)[:_MAX_PARAM_LEN]
    return out


class MlflowTracker:
    """Thin, failure-proof wrapper. Only the master rank should call `start`."""

    def __init__(self) -> None:
        self.enabled = False
        self._mlflow = None
        self._ui_proc: Optional[subprocess.Popen] = None
        self.uri: Optional[str] = None

    def start(
        self,
        *,
        tracking_uri: Optional[str],
        experiment_name: str,
        run_name: str,
        run_id: Optional[str],
        params: dict[str, Any],
        tags: Optional[dict[str, str]] = None,
        log_system_metrics: bool = False,
        artifacts: Optional[list[str]] = None,
    ) -> bool:
        try:
            import mlflow
        except Exception:
            LOG.warning("mlflow_unavailable", hint="pip install mlflow")
            return False

        try:
            uri = os.environ.get("MLFLOW_TRACKING_URI") or tracking_uri or "mlruns"
            # Recent MLflow releases refuse plain-directory ("file") stores unless
            # explicitly allowed. We want them: no server or DB needed, and a run
            # dir can just be rsync'd/`modal volume get`-ed off the box and browsed
            # with `mlflow ui`. Point tracking_uri at sqlite:///... or http://... to
            # use a real backend instead.
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment_name)

            start_kwargs: dict[str, Any] = {"run_id": run_id} if run_id else {"run_name": run_name}
            if log_system_metrics:
                try:
                    run = mlflow.start_run(log_system_metrics=True, **start_kwargs)
                except TypeError:  # mlflow < 2.8 has no system metrics
                    run = mlflow.start_run(**start_kwargs)
            else:
                run = mlflow.start_run(**start_kwargs)

            if not run_id:  # params of a resumed run are already recorded (and immutable)
                mlflow.log_params(_flatten(params))
            if tags:
                mlflow.set_tags(tags)
            for path in artifacts or []:
                if os.path.isfile(path):
                    mlflow.log_artifact(path)

            self._mlflow = mlflow
            self.enabled = True
            self.uri = uri
            LOG.info("mlflow_started", uri=uri, experiment=experiment_name, run_id=run.info.run_id)
            return True
        except Exception as exc:
            LOG.warning("mlflow_init_failed", error=str(exc))
            self.enabled = False
            return False

    def log_metrics(self, metrics: dict[str, Optional[float]], step: int) -> None:
        if not self.enabled:
            return
        clean = {
            k: float(v) for k, v in metrics.items()
            if v is not None and math.isfinite(float(v))
        }
        if not clean:
            return
        try:
            self._mlflow.log_metrics(clean, step=step)
        except Exception as exc:
            LOG.warning("mlflow_log_failed", error=str(exc))

    def start_ui(self, host: str = "0.0.0.0", port: int = 5000) -> bool:
        """Serve the MLflow UI from a background subprocess for the life of the run.

        Only meaningful for a local store (directory / sqlite): with an http(s)
        tracking URI there's already a server, so this is skipped. Stopped in
        `end()` and at interpreter exit.
        """
        if not self.enabled or not self.uri or self.uri.startswith(("http://", "https://")):
            return False
        if self._ui_proc is not None:
            return True
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                LOG.warning("mlflow_ui_port_in_use", port=port, hint="set mlflow.ui.port or MLFLOW_UI_PORT")
                return False
        try:
            self._ui_proc = subprocess.Popen(
                [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", self.uri,
                 "--host", host, "--port", str(port)],
                env={**os.environ, "MLFLOW_ALLOW_FILE_STORE": "true"},
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            atexit.register(self._stop_ui)
            LOG.info("mlflow_ui_started", host=host, port=port, store=self.uri)
            return True
        except Exception as exc:
            LOG.warning("mlflow_ui_failed", error=str(exc))
            return False

    def _stop_ui(self) -> None:
        if self._ui_proc is not None and self._ui_proc.poll() is None:
            self._ui_proc.terminate()
        self._ui_proc = None

    def end(self, status: str = "FINISHED") -> None:
        self._stop_ui()
        if not self.enabled:
            return
        try:
            self._mlflow.end_run(status=status)
        except Exception as exc:
            LOG.warning("mlflow_end_failed", error=str(exc))
        self.enabled = False
