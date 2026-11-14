import csv
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader, TensorDataset
import wandb


@pytest.fixture
def runtime_loggers(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WANDB_MODE", "disabled")

    def capture(namespace, task, checkpoint=None):
        module = importlib.import_module(f"{namespace}.{task}")
        model = SimpleNamespace(
            model=torch.nn.Linear(1, 1),
            finetune_status={},
            finetune_hparams=lambda: {},
            finetune_param_group_rows=lambda: [],
        )
        bundle = SimpleNamespace(
            model=SimpleNamespace(channels=[], backbone=SimpleNamespace(moe=None)),
            loss=None,
            averaging=None,
            finetune=None,
            data=SimpleNamespace(mask_rate=0.15, max_tokens=2),
            adapt=SimpleNamespace(new_channels=["eeg"]),
        )
        args = SimpleNamespace(
            config=Path("config.yaml"),
            version="unit",
            version_name="unit",
            backbone_arch="test",
            ckpt_path=checkpoint,
            pretrained_backbone_path=None,
            phase="stage1",
            channel_names=["eeg"],
            strategy="auto",
            devices=[0],
            epochs=2,
            patience=10,
            monitor="val_cohen_kappa",
            monitor_mod="max",
            ckpt_every_n_epochs=30,
            gradient_clip_val=0.0,
            accumulate_grad_batches=1,
            precision="32-true",
            check_val_every_n_epoch=1,
            print_diagnostics=False,
            test_after_fit=False,
            test_all_checkpoints_after_fit=False,
            label_name="stage5",
            wandb_mode="disabled",
            train_pair_monitor_enable=False,
            train_pair_monitor_log_prefix="pairs",
            train_pair_skew_warn_threshold=1.0,
            train_pair_min_unique_coverage_warn_threshold=0.0,
        )
        monkeypatch.setattr(
            module,
            "persist_run_config_and_args",
            lambda args, path, **kwargs: Path(path).mkdir(parents=True, exist_ok=True),
        )
        if task == "finetune":
            monkeypatch.setattr(module, "prepare_dataloader", lambda args: ([], [], None))
            monkeypatch.setattr(module, "Sleep2vecFinetuning", lambda *args, **kwargs: model)
            monkeypatch.setattr(module, "save_training_run_manifest", lambda *args, **kwargs: None)
            monkeypatch.setattr(module, "_finish_wandb_run", lambda *args: None)
            monkeypatch.setattr(module, "DDPStrategy", lambda **kwargs: "auto")
        else:
            monkeypatch.setattr(module, "load_pretrain_config", lambda path: bundle)
            monkeypatch.setattr(module, "apply_model_config_args", lambda *args, **kwargs: None)
            monkeypatch.setattr(module, "apply_data_backend_args", lambda *args, **kwargs: None)
            monkeypatch.setattr(module, "get_pretrain_dataloader", lambda args: ([], []))
            monkeypatch.setattr(module, "handles_distributed_sharding", lambda sampler: False)
            monkeypatch.setattr(module, "PairAccLoggerCallback", lambda *args, **kwargs: pl.Callback())
            factory = "Sleep2vecPretraining" if task == "pretrain" else "Sleep2vecAdaptation"
            monkeypatch.setattr(module, factory, lambda *args, **kwargs: model)
            if task == "adapt":
                monkeypatch.setattr(module, "initial_pair_probs_for_phase", lambda *args, **kwargs: {})
            if namespace == "sleep2expert" and task == "pretrain":
                for name in (
                    "count_total_parameters",
                    "count_trainable_parameters",
                    "estimate_active_parameters_per_token",
                    "estimate_moe_ffn_active_flops",
                    "estimate_dense_equivalent_ffn_flops",
                ):
                    monkeypatch.setattr(module, name, lambda *args, **kwargs: 1)

        captured = {}

        def trainer(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("capture local metric loggers")

        monkeypatch.setattr(module, "pl", SimpleNamespace(Trainer=trainer))
        with pytest.raises(RuntimeError, match="capture local metric loggers"):
            if task == "finetune":
                module.supervised(args, bundle)
            else:
                getattr(module, f"sleep2vec_{task}")(args)
        return captured["logger"]

    yield capture
    wandb.finish()


class EpochMetricsModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.observed_before_next_epoch = []
        self.started_epoch = False

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):
        loss = (self.weight - batch[0]).square().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.log("val_cohen_kappa", 0.5 - self.current_epoch * 0.1, on_step=False, on_epoch=True)

    def on_train_epoch_start(self):
        if not self.started_epoch:
            self.started_epoch = True
            return
        logger = next(logger for logger in self.trainer.loggers if isinstance(logger, CSVLogger))
        rows = list(csv.DictReader((Path(logger.log_dir) / "metrics.csv").open()))
        previous = [row for row in rows if row.get("epoch") == str(self.current_epoch - 1)]
        assert any(row.get("val_cohen_kappa") for row in previous)
        assert any(row.get("train_loss") for row in previous)
        self.observed_before_next_epoch.append(self.current_epoch - 1)


def fit_with_loggers(loggers, epochs, checkpoint=None):
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=loggers,
        max_epochs=epochs,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
    )
    model = EpochMetricsModel()
    loader = DataLoader(TensorDataset(torch.zeros(2, 1)), batch_size=1)
    trainer.fit(model, loader, loader, ckpt_path=checkpoint)
    return trainer, model


@pytest.mark.parametrize("namespace", ["sleep2vec", "sleep2vec2", "sleep2expert"])
@pytest.mark.parametrize("task", ["pretrain", "adapt", "finetune"])
def test_runtime_persists_each_epoch_with_wandb_disabled(runtime_loggers, namespace, task):
    loggers = runtime_loggers(namespace, task)
    trainer, model = fit_with_loggers(loggers, 2)
    assert model.observed_before_next_epoch == [0]
    logger = next(logger for logger in trainer.loggers if isinstance(logger, CSVLogger))
    rows = list(csv.DictReader((Path(logger.log_dir) / "metrics.csv").open()))
    validation = [row for row in rows if row.get("val_cohen_kappa")]
    assert [int(row["epoch"]) for row in validation] == [0, 1]
    assert [float(row["val_cohen_kappa"]) for row in validation] == pytest.approx([0.5, 0.4])
    assert [int(row["step"]) for row in validation] == [1, 3]


@pytest.mark.parametrize("namespace", ["sleep2vec", "sleep2vec2", "sleep2expert"])
def test_pretrain_resume_preserves_previous_local_metric_history(runtime_loggers, namespace):
    loggers = runtime_loggers(namespace, "pretrain")
    trainer, _ = fit_with_loggers(loggers, 2)
    logger = next(logger for logger in trainer.loggers if isinstance(logger, CSVLogger))
    previous = Path(logger.log_dir) / "metrics.csv"
    original = previous.read_bytes()
    checkpoint = previous.parents[2] / "checkpoints" / "resume.ckpt"
    trainer.save_checkpoint(checkpoint)
    wandb.finish()
    resumed_loggers = runtime_loggers(namespace, "pretrain", checkpoint)
    resumed_trainer, _ = fit_with_loggers(resumed_loggers, 3, checkpoint)
    resumed_logger = next(logger for logger in resumed_trainer.loggers if isinstance(logger, CSVLogger))
    assert Path(resumed_logger.log_dir) != Path(logger.log_dir)
    assert previous.read_bytes() == original
    rows = list(csv.DictReader((Path(resumed_logger.log_dir) / "metrics.csv").open()))
    validation = [row for row in rows if row.get("val_cohen_kappa")]
    assert [int(row["epoch"]) for row in validation] == [2]
    assert float(validation[0]["val_cohen_kappa"]) == pytest.approx(0.3)
