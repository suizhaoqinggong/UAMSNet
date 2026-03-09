from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias, Any, Literal, cast

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch.cuda.amp import GradScaler

from framework.contracts import Batch, Metric, MetricResults, ModelAdapter, Runner, Task
from framework.logging import (
    ExperimentLogger,
    NullExperimentLogger,
    flatten_mapping,
)

from .checkpoint import (
    CheckpointManager,
    CheckpointSettings,
    extract_best_monitor_value,
    extract_epoch,
    extract_optional_state_mapping,
    extract_state_mapping,
    load_checkpoint_payload,
)
from .device import (
    autocast_context,
    build_grad_scaler,
    move_batch_to_device,
    resolve_device,
)
from .evaluator import (
    Evaluator,
    collect_metric_results,
    reset_metrics,
    set_model_mode,
    to_float,
    update_metrics,
)
from .seed import set_global_seed

OptimizerLike: TypeAlias = torch.optim.Optimizer

CONFUSION_CLASS_PATTERN = re.compile(r"^cm_(?P<label>.+)_(?P<field>tn|fp|fn|tp)$")
ANSI_RESET = "\033[0m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"
DEFAULT_TENSORBOARD_SCALAR_TAGS = (
    "train/loss",
    "val/loss",
    "val/accuracy",
    "val/auroc",
    "val/auprc",
    "val/precision",
    "val/recall",
    "val/f1",
    "val/clinical_balanced_accuracy",
)


@dataclass(frozen=True, slots=True)
class TrainerSettings:
    epochs: int
    run_dir: str
    checkpoints_dir: str
    tensorboard_dir: str
    logs_dir: str
    confusion_matrix_dir: str
    tensorboard_enabled: bool = True
    tensorboard_flush_secs: int = 30
    tensorboard_max_queue: int = 10
    histogram_every_n_epochs: int = 1
    confusion_matrix_every_n_epochs: int = 1
    tensorboard_scalar_tags: tuple[str, ...] = DEFAULT_TENSORBOARD_SCALAR_TAGS
    log_interval_steps: int = 20
    patience: int | None = None
    device: str | None = "auto"
    amp: bool = False
    grad_clip_norm: float | None = None
    seed: int | None = None
    deterministic: bool = True
    checkpoint_monitor: str = "loss"
    checkpoint_mode: Literal["min", "max"] = "min"
    resume_from: str | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than zero.")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be greater than zero when set.")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative when set.")
        if not self.run_dir.strip():
            raise ValueError("run_dir cannot be empty.")
        if not self.checkpoints_dir.strip():
            raise ValueError("checkpoints_dir cannot be empty.")
        if not self.tensorboard_dir.strip():
            raise ValueError("tensorboard_dir cannot be empty.")
        if not self.logs_dir.strip():
            raise ValueError("logs_dir cannot be empty.")
        if not self.confusion_matrix_dir.strip():
            raise ValueError("confusion_matrix_dir cannot be empty.")
        if not self.checkpoint_monitor.strip():
            raise ValueError("checkpoint_monitor cannot be empty.")
        if self.tensorboard_flush_secs <= 0:
            raise ValueError("tensorboard_flush_secs must be greater than zero.")
        if self.tensorboard_max_queue <= 0:
            raise ValueError("tensorboard_max_queue must be greater than zero.")
        if self.histogram_every_n_epochs <= 0:
            raise ValueError("histogram_every_n_epochs must be greater than zero.")
        if self.confusion_matrix_every_n_epochs <= 0:
            raise ValueError("confusion_matrix_every_n_epochs must be greater than zero.")
        if self.log_interval_steps <= 0:
            raise ValueError("log_interval_steps must be greater than zero.")
        if self.patience is not None and self.patience <= 0:
            raise ValueError("patience must be greater than zero when set.")
        for tag in self.tensorboard_scalar_tags:
            if not tag.strip():
                raise ValueError("tensorboard_scalar_tags cannot contain empty entries.")


class Trainer(Runner):
    def __init__(
        self,
        *,
        model: ModelAdapter,
        task: Task,
        optimizer: OptimizerLike,
        train_batches: Iterable[Batch],
        val_batches: Iterable[Batch],
        train_metrics: Sequence[Metric],
        val_metrics: Sequence[Metric],
        settings: TrainerSettings,
        test_batches: Iterable[Batch] | None = None,
        test_metrics: Sequence[Metric] | None = None,
        config_snapshot: Mapping[str, object] | None = None,
        experiment_logger: ExperimentLogger | None = None,
    ) -> None:
        self._model = model
        self._task = task
        self._optimizer = optimizer
        self._train_batches = train_batches
        self._val_batches = val_batches
        self._test_batches = test_batches
        self._train_metrics = list(train_metrics)
        self._settings = settings
        self._config_snapshot = {} if config_snapshot is None else dict(config_snapshot)
        self._logger = experiment_logger or build_logger_from_config(
            self._config_snapshot,
            settings=settings,
        )
        self._run_active = False
        self._params_logged = False
        self._last_completed_epoch: int | None = None

        self._run_dir = Path(settings.run_dir)
        self._checkpoints_dir = Path(settings.checkpoints_dir)
        self._tensorboard_dir = Path(settings.tensorboard_dir)
        self._logs_dir = Path(settings.logs_dir)
        self._confusion_matrix_dir = Path(settings.confusion_matrix_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._confusion_matrix_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_jsonl_path = self._logs_dir / "metrics.jsonl"
        self._summary_path = self._logs_dir / "summary.json"

        self._device = resolve_device(settings.device)
        self._grad_scaler = build_grad_scaler(enabled=settings.amp, device=self._device)
        self._evaluator = Evaluator(
            task=task,
            metrics=val_metrics,
            device=self._device,
            amp_enabled=settings.amp,
        )
        self._test_evaluator = (
            Evaluator(
                task=task,
                metrics=test_metrics,
                device=self._device,
                amp_enabled=settings.amp,
            )
            if test_metrics is not None
            else None
        )

        checkpoint_settings = CheckpointSettings(
            directory=settings.checkpoints_dir,
            monitor=settings.checkpoint_monitor,
            mode=settings.checkpoint_mode,
        )
        self._checkpoint_manager = CheckpointManager(checkpoint_settings)

        self._start_epoch = 0

        if settings.seed is not None:
            set_global_seed(settings.seed, deterministic=settings.deterministic)

        self._model.to(self._device)

        if settings.resume_from is not None:
            self.resume_from_checkpoint(settings.resume_from)

    def train(self) -> MetricResults:
        return self._run_with_logging(self._train_impl, command="train")

    def validate(self, checkpoint_path: str | None = None) -> MetricResults:
        return self._run_with_logging(
            lambda: self._validate_impl(checkpoint_path),
            command="validate",
        )

    def test(self, checkpoint_path: str | None = None) -> MetricResults:
        return self._run_with_logging(
            lambda: self._test_impl(checkpoint_path),
            command="test",
        )

    def fit(self) -> MetricResults:
        return self._run_with_logging(self._fit_impl, command="fit")

    def train_one_epoch(self, *, epoch: int) -> MetricResults:
        set_model_mode(self._model, training=True)
        reset_metrics(self._train_metrics)

        total_loss = 0.0
        batch_count = 0
        total_batches = infer_num_batches(self._train_batches)

        for step, batch in enumerate(self._train_batches, start=1):
            device_batch = move_batch_to_device(batch, device=self._device)
            if step == 1:
                image_shape = tuple(device_batch["image"].shape)
                label_shape = tuple(device_batch["label"].shape)
                self._log_state(
                    "DATA",
                    (
                        f"train_batch image_shape={image_shape} "
                        f"label_shape={label_shape} image_device={device_batch['image'].device}"
                    ),
                    color=ANSI_CYAN,
                )
            zero_grad(self._optimizer)

            with autocast_context(enabled=self._settings.amp, device=self._device):
                outputs = self._model.forward(device_batch)
                loss = self._task.compute_loss(outputs, device_batch)

            backward_and_step(
                loss=loss,
                optimizer=self._optimizer,
                grad_scaler=self._grad_scaler,
                model=self._model,
                grad_clip_norm=self._settings.grad_clip_norm,
            )

            predictions = self._task.postprocess_outputs(outputs)
            targets = self._task.extract_targets(device_batch)
            update_metrics(self._train_metrics, predictions, targets)

            loss_value = to_float(loss)
            total_loss += loss_value
            batch_count = step

            if should_log_batch_progress(
                step=step,
                total_steps=total_batches,
                interval_steps=self._settings.log_interval_steps,
            ):
                self._log_state(
                    "BATCH",
                    build_batch_summary(
                        epoch=epoch,
                        total_epochs=self._settings.epochs,
                        step=step,
                        total_steps=total_batches,
                        loss_value=loss_value,
                        avg_loss=(total_loss / batch_count),
                    ),
                    color=ANSI_YELLOW,
                )

        if batch_count == 0:
            raise ValueError("Trainer received zero training batches.")

        metrics = collect_metric_results(self._train_metrics)
        metrics["loss"] = total_loss / batch_count
        return metrics

    def load_model_from_checkpoint(self, checkpoint_path: str) -> None:
        payload = load_checkpoint_payload(checkpoint_path, map_location=self._device)
        model_state = extract_state_mapping(payload, "model_state")
        self._model.load_state_dict(cast(Mapping[str, object], model_state))

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        payload = load_checkpoint_payload(checkpoint_path, map_location=self._device)

        model_state = extract_state_mapping(payload, "model_state")
        self._model.load_state_dict(cast(Mapping[str, object], model_state))

        optimizer_state = extract_state_mapping(payload, "optimizer_state")
        self._optimizer.load_state_dict(dict(optimizer_state))

        if self._grad_scaler is not None:
            scaler_state = extract_optional_state_mapping(payload, "scaler_state")
            if scaler_state is not None:
                self._grad_scaler.load_state_dict(dict(scaler_state))

        resumed_epoch = extract_epoch(payload)
        self._start_epoch = resumed_epoch + 1
        if self._start_epoch >= self._settings.epochs:
            raise ValueError(
                "Resumed epoch is already at or beyond target training epochs. "
                f"start_epoch={self._start_epoch}, epochs={self._settings.epochs}."
            )

        best_monitor_value = extract_best_monitor_value(payload)
        if best_monitor_value is not None:
            self._checkpoint_manager.update_best_state(
                best_monitor_value=best_monitor_value,
                best_epoch=resumed_epoch,
            )

    def _run_with_logging(
        self,
        operation: Callable[[], MetricResults],
        *,
        command: str,
    ) -> MetricResults:
        self._log_state(
            "STATE",
            (
                f"command={command} device={self._device.type} amp={self._settings.amp} "
                f"epochs={self._settings.epochs} patience={self._settings.patience} "
                f"run_dir={self._run_dir}"
            ),
            color=ANSI_CYAN,
        )
        self._start_logging_run()
        try:
            result = operation()
        except Exception as exc:
            self._log_state(
                "ERROR",
                f"command={command} failed: {exc.__class__.__name__}: {exc}",
                color=ANSI_RED,
            )
            self._write_summary(status="FAILED", final_metrics=None)
            self._end_logging_run(status="FAILED")
            raise

        self._write_summary(status="FINISHED", final_metrics=result)
        self._end_logging_run(status="FINISHED")
        self._log_state(
            "DONE",
            f"command={command} finished summary={self._summary_path}",
            color=ANSI_GREEN,
        )
        return result

    def _train_impl(self) -> MetricResults:
        last_train_result: MetricResults | None = None
        last_val_result: MetricResults | None = None
        stale_epochs = 0
        self._last_completed_epoch = None

        for epoch in range(self._start_epoch, self._settings.epochs):
            self._log_state(
                "EPOCH",
                f"start epoch={epoch + 1}/{self._settings.epochs}",
                color=ANSI_BLUE,
            )
            last_train_result = self.train_one_epoch(epoch=epoch)
            last_val_result = self._validate_impl(None, log_metrics=False)

            epoch_metrics = with_prefix(last_train_result, "train") | with_prefix(
                last_val_result,
                "val",
            )
            self._logger.log_metrics(
                select_tensorboard_metrics(
                    metrics=epoch_metrics,
                    allowed_tags=self._settings.tensorboard_scalar_tags,
                ),
                step=epoch,
            )
            self._append_metrics_event(event="epoch", metrics=epoch_metrics, step=epoch)

            improved = self._save_checkpoint_if_enabled(
                epoch=epoch,
                train_metrics=last_train_result,
                val_metrics=last_val_result,
            )
            if improved:
                stale_epochs = 0
                self._log_state(
                    "CKPT",
                    f"best checkpoint updated at {self._checkpoint_manager.best_path}",
                    color=ANSI_MAGENTA,
                )
            else:
                stale_epochs += 1

            self._maybe_log_model_histograms(epoch=epoch)
            self._maybe_log_confusion_matrix(
                split="val",
                metrics=last_val_result,
                step=epoch,
            )
            self._log_state(
                "EPOCH",
                build_epoch_summary(
                    epoch=epoch,
                    total_epochs=self._settings.epochs,
                    train_metrics=last_train_result,
                    val_metrics=last_val_result,
                ),
                color=ANSI_GREEN,
            )
            self._last_completed_epoch = epoch

            if self._settings.patience is not None and stale_epochs >= self._settings.patience:
                self._log_state(
                    "EARLY_STOP",
                    (
                        f"epoch={epoch + 1}/{self._settings.epochs} "
                        f"monitor=val/{self._settings.checkpoint_monitor} "
                        f"patience={self._settings.patience}"
                    ),
                    color=ANSI_MAGENTA,
                )
                break

        if last_train_result is None or last_val_result is None:
            raise RuntimeError("Training loop did not produce results.")

        self._log_checkpoint_artifacts()
        combined = with_prefix(last_train_result, "train") | with_prefix(last_val_result, "val")
        self._append_metrics_event(
            event="train_complete",
            metrics=combined,
            step=(
                self._last_completed_epoch
                if self._last_completed_epoch is not None
                else self._settings.epochs - 1
            ),
        )
        return combined

    def _validate_impl(
        self,
        checkpoint_path: str | None,
        *,
        log_metrics: bool = True,
    ) -> MetricResults:
        if checkpoint_path is not None:
            self.load_model_from_checkpoint(checkpoint_path)
        metrics = self._evaluator.evaluate(model=self._model, batches=self._val_batches)
        if log_metrics:
            prefixed = with_prefix(metrics, "val")
            self._logger.log_metrics(
                select_tensorboard_metrics(
                    metrics=prefixed,
                    allowed_tags=self._settings.tensorboard_scalar_tags,
                )
            )
            self._append_metrics_event(event="validate", metrics=prefixed, step=None)
            self._maybe_log_confusion_matrix(split="val", metrics=metrics, step=None)
            self._log_state(
                "VALIDATE",
                build_split_summary(split="val", metrics=metrics),
                color=ANSI_YELLOW,
            )
        return metrics

    def _test_impl(
        self,
        checkpoint_path: str | None,
        *,
        log_metrics: bool = True,
    ) -> MetricResults:
        if checkpoint_path is not None:
            self.load_model_from_checkpoint(checkpoint_path)
        if self._test_batches is None:
            raise RuntimeError("test_batches is not configured.")
        if self._test_evaluator is None:
            raise RuntimeError("test_metrics is not configured.")

        metrics = self._test_evaluator.evaluate(model=self._model, batches=self._test_batches)
        if log_metrics:
            prefixed = with_prefix(metrics, "test")
            self._append_metrics_event(event="test", metrics=prefixed, step=None)
            self._log_test_summary(metrics=metrics)
            self._log_state(
                "TEST",
                build_split_summary(split="test", metrics=metrics),
                color=ANSI_YELLOW,
            )
        return metrics

    def _fit_impl(self) -> MetricResults:
        self._log_state("FIT", "phase=train", color=ANSI_BLUE)
        results = self._train_impl()
        if self._test_batches is None or self._test_evaluator is None:
            return results
        best_checkpoint = self._checkpoint_manager.best_path
        checkpoint_path = str(best_checkpoint) if best_checkpoint.exists() else None
        if checkpoint_path is None:
            self._log_state("FIT", "phase=test checkpoint=current_weights", color=ANSI_BLUE)
        else:
            self._log_state("FIT", f"phase=test checkpoint={checkpoint_path}", color=ANSI_BLUE)
        test_results = self._test_impl(checkpoint_path)
        combined = results | with_prefix(test_results, "test")
        self._append_metrics_event(
            event="fit_complete",
            metrics=combined,
            step=(
                self._last_completed_epoch
                if self._last_completed_epoch is not None
                else self._settings.epochs - 1
            ),
        )
        return combined

    def _start_logging_run(self) -> None:
        if not self._run_active:
            self._logger.start_run(run_name=self._infer_run_name())
            self._run_active = True
            self._params_logged = False
        if not self._params_logged:
            self._logger.log_params(flatten_mapping(self._config_snapshot))
            self._params_logged = True

    def _end_logging_run(self, *, status: str) -> None:
        if not self._run_active:
            return
        self._logger.end_run(status=status)
        self._run_active = False
        self._params_logged = False

    def _infer_run_name(self) -> str | None:
        experiment = self._config_snapshot.get("experiment")
        if not isinstance(experiment, Mapping):
            return None
        name = experiment.get("name")
        if isinstance(name, str):
            stripped_name = name.strip()
            if stripped_name:
                return stripped_name
        return None

    def _save_checkpoint_if_enabled(
        self,
        *,
        epoch: int,
        train_metrics: Mapping[str, float],
        val_metrics: Mapping[str, float],
    ) -> bool:
        model_state = export_mapping_state_dict(self._model.state_dict(), component_name="model")
        optimizer_state = export_mapping_state_dict(
            self._optimizer.state_dict(),
            component_name="optimizer",
        )
        scaler_state = (
            export_mapping_state_dict(self._grad_scaler.state_dict(), component_name="grad_scaler")
            if self._grad_scaler is not None
            else None
        )

        return self._checkpoint_manager.maybe_save(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            model_state=model_state,
            optimizer_state=optimizer_state,
            scaler_state=scaler_state,
            config_snapshot=self._config_snapshot,
        )

    def _log_checkpoint_artifacts(self) -> None:
        best_path = self._checkpoint_manager.best_path
        if best_path.exists():
            self._logger.log_artifact(str(best_path))

        last_path = self._checkpoint_manager.last_path
        if last_path.exists():
            self._logger.log_artifact(str(last_path))

    def _append_metrics_event(
        self,
        *,
        event: str,
        metrics: Mapping[str, float],
        step: int | None,
    ) -> None:
        payload: dict[str, object] = {
            "timestamp_utc": datetime.now(tz=UTC).isoformat(),
            "event": event,
            "step": step,
            "metrics": {key: float(value) for key, value in metrics.items()},
        }
        with self._metrics_jsonl_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(payload, sort_keys=True) + "\n")

    def _log_test_summary(self, *, metrics: Mapping[str, float]) -> None:
        summary_content = build_metrics_table_markdown(
            metrics=metrics,
        )
        summary_path = self._logs_dir / "test_summary.md"
        summary_path.write_text(summary_content, encoding="utf-8")
        self._logger.log_text("test/summary", summary_content, step=0)
        self._logger.log_artifact(str(summary_path))

    def _log_state(self, level: str, message: str, *, color: str) -> None:
        timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
        formatted_level = paint(level, color)
        stream = sys.stdout
        stream.write(f"{paint(f'[{timestamp}]', ANSI_CYAN)} {formatted_level} {message}\n")
        stream.flush()

    def _write_summary(
        self,
        *,
        status: str,
        final_metrics: Mapping[str, float] | None,
    ) -> None:
        summary: dict[str, object] = {
            "status": status,
            "timestamp_utc": datetime.now(tz=UTC).isoformat(),
            "run": {
                "run_dir": str(self._run_dir),
                "checkpoints_dir": str(self._checkpoints_dir),
                "tensorboard_dir": str(self._tensorboard_dir),
                "logs_dir": str(self._logs_dir),
                "confusion_matrix_dir": str(self._confusion_matrix_dir),
            },
            "metrics": (
                {}
                if final_metrics is None
                else {key: float(value) for key, value in final_metrics.items()}
            ),
        }
        self._summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _maybe_log_model_histograms(self, *, epoch: int) -> None:
        if not self._settings.tensorboard_enabled:
            return
        if not should_log_epoch(
            epoch=epoch,
            every_n_epochs=self._settings.histogram_every_n_epochs,
        ):
            return
        for name, parameter in iter_named_parameters(self._model):
            self._logger.log_histogram(
                f"params/{name}",
                parameter.detach().float().cpu(),
                step=epoch,
            )
            if parameter.grad is None:
                continue
            self._logger.log_histogram(
                f"grads/{name}",
                parameter.grad.detach().float().cpu(),
                step=epoch,
            )

    def _maybe_log_confusion_matrix(
        self,
        *,
        split: str,
        metrics: Mapping[str, float],
        step: int | None,
    ) -> None:
        if step is not None and not should_log_epoch(
            epoch=step,
            every_n_epochs=self._settings.confusion_matrix_every_n_epochs,
        ):
            return

        per_class = collect_class_confusion(metrics)
        if not per_class:
            return

        figure = build_confusion_matrix_figure(per_class=per_class, split=split)
        output_name = (
            f"{split}_final.png"
            if step is None
            else f"{split}_epoch_{step:04d}.png"
        )
        output_path = self._confusion_matrix_dir / output_name
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        self._logger.log_figure(f"confusion_matrix/{split}", figure, step=step)
        self._logger.log_artifact(str(output_path))
        plt.close(figure)


def zero_grad(optimizer: OptimizerLike) -> None:
    optimizer.zero_grad(set_to_none=True)


def backward_and_step(
    *,
    loss: torch.Tensor,
    optimizer: OptimizerLike,
    grad_scaler: GradScaler | None,
    model: ModelAdapter,
    grad_clip_norm: float | None,
) -> None:
    if grad_scaler is None:
        torch.autograd.backward(loss)
        maybe_clip_gradients(model=model, max_norm=grad_clip_norm)
        optimizer.step()
        return

    scaled_loss = cast(torch.Tensor, grad_scaler.scale(loss))
    torch.autograd.backward(scaled_loss)
    if grad_clip_norm is not None:
        grad_scaler.unscale_(optimizer)
    maybe_clip_gradients(model=model, max_norm=grad_clip_norm)
    grad_scaler.step(optimizer)
    grad_scaler.update()


def maybe_clip_gradients(*, model: ModelAdapter, max_norm: float | None) -> None:
    if max_norm is None:
        return
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def export_mapping_state_dict(
    state_dict: Mapping[str, object],
    *,
    component_name: str,
) -> dict[str, object]:
    validated: dict[str, object] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError(f"{component_name}.state_dict() must use string keys.")
        validated[key] = value
    return validated


def with_prefix(metrics: Mapping[str, float], prefix: str) -> MetricResults:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def paint(text: str, color: str) -> str:
    return f"{color}{text}{ANSI_RESET}"


def build_epoch_summary(
    *,
    epoch: int,
    total_epochs: int,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
) -> str:
    train_loss = format_metric(train_metrics, "loss")
    val_loss = format_metric(val_metrics, "loss")
    val_auc = format_metric(val_metrics, "auroc")
    val_f1 = format_metric(val_metrics, "f1")
    return (
        f"done epoch={epoch + 1}/{total_epochs} "
        f"train_loss={train_loss} val_loss={val_loss} val_auroc={val_auc} val_f1={val_f1}"
    )


def build_split_summary(*, split: str, metrics: Mapping[str, float]) -> str:
    loss = format_metric(metrics, "loss")
    auc = format_metric(metrics, "auroc")
    f1 = format_metric(metrics, "f1")
    accuracy = format_metric(metrics, "accuracy")
    return f"{split} loss={loss} auroc={auc} f1={f1} accuracy={accuracy}"


def format_metric(metrics: Mapping[str, float], key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def select_tensorboard_metrics(
    *,
    metrics: Mapping[str, float],
    allowed_tags: Sequence[str],
) -> dict[str, float]:
    allowed = {tag.strip() for tag in allowed_tags if tag.strip()}
    if not allowed:
        return {}
    selected: dict[str, float] = {}
    for key, value in metrics.items():
        if key in allowed:
            selected[key] = float(value)
    return selected


def build_metrics_table_markdown(
    *,
    metrics: Mapping[str, float],
) -> str:
    lines = ["| Metric | Value |", "| --- | ---: |"]
    for key in sorted(metrics):
        lines.append(f"| `{key}` | {format_markdown_metric_value(float(metrics[key]))} |")
    return "\n".join(lines) + "\n"


def format_markdown_metric_value(value: float) -> str:
    if value != value:
        return "NaN"
    if value == float("inf"):
        return "Infinity"
    if value == float("-inf"):
        return "-Infinity"
    return f"{value:.6f}"


def should_log_epoch(*, epoch: int, every_n_epochs: int) -> bool:
    return epoch % every_n_epochs == 0


def infer_num_batches(batches: Iterable[Batch]) -> int | None:
    if not isinstance(batches, Sized):
        return None
    return len(batches)


def should_log_batch_progress(
    *,
    step: int,
    total_steps: int | None,
    interval_steps: int,
) -> bool:
    if step == 1:
        return True
    if total_steps is not None and step == total_steps:
        return True
    return step % interval_steps == 0


def build_batch_summary(
    *,
    epoch: int,
    total_epochs: int,
    step: int,
    total_steps: int | None,
    loss_value: float,
    avg_loss: float,
) -> str:
    if total_steps is None:
        step_text = f"{step}"
    else:
        step_text = f"{step}/{total_steps}"
    return (
        f"epoch={epoch + 1}/{total_epochs} "
        f"step={step_text} loss={loss_value:.6f} avg_loss={avg_loss:.6f}"
    )


def iter_named_parameters(model: ModelAdapter) -> list[tuple[str, torch.nn.Parameter]]:
    if isinstance(model, torch.nn.Module):
        return list(model.named_parameters())
    return [(f"param_{index:04d}", parameter) for index, parameter in enumerate(model.parameters())]


def collect_class_confusion(metrics: Mapping[str, float]) -> dict[str, dict[str, float]]:
    per_class: dict[str, dict[str, float]] = {}
    for key, value in metrics.items():
        match = CONFUSION_CLASS_PATTERN.match(key)
        if match is None:
            continue
        label = match.group("label")
        field = match.group("field")
        class_metrics = per_class.setdefault(label, {})
        class_metrics[field] = float(value)
    complete: dict[str, dict[str, float]] = {}
    for label, values in per_class.items():
        if {"tn", "fp", "fn", "tp"}.issubset(values):
            complete[label] = values
    return complete


def build_confusion_matrix_figure(
    *,
    per_class: Mapping[str, Mapping[str, float]],
    split: str,
) -> Figure:
    labels = sorted(per_class)
    columns = min(3, len(labels))
    rows = (len(labels) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 4.0 * rows))

    if isinstance(axes, Axes):
        axes_list = [axes]
    else:
        axes_list = [cast(Axes, axis) for axis in cast(Any, axes).flat]

    for axis, label in zip(axes_list, labels, strict=False):
        matrix = [
            [per_class[label]["tn"], per_class[label]["fp"]],
            [per_class[label]["fn"], per_class[label]["tp"]],
        ]
        image = axis.imshow(matrix, cmap="Blues")
        axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title(label)
        axis.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        axis.set_yticks([0, 1], labels=["True 0", "True 1"])

        for row_index in range(2):
            for column_index in range(2):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index][column_index]:.0f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )

    for axis in axes_list[len(labels) :]:
        axis.axis("off")

    figure.suptitle(f"{split.upper()} multilabel confusion matrix", fontsize=14)
    figure.tight_layout()
    return figure


def build_logger_from_config(
    config: Mapping[str, object],
    *,
    settings: TrainerSettings,
) -> ExperimentLogger:
    del config
    if not settings.tensorboard_enabled:
        return NullExperimentLogger()
    try:
        from framework.logging.tensorboard_logger import TensorBoardLogger
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorBoard logging is enabled but 'tensorboard' is not installed in this environment."
        ) from exc

    return TensorBoardLogger(
        log_dir=settings.tensorboard_dir,
        flush_secs=settings.tensorboard_flush_secs,
        max_queue=settings.tensorboard_max_queue,
    )
