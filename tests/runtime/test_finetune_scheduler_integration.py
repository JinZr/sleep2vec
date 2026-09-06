from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture(params=["sleep2vec", "sleep2vec2", "sleep2expert"])
def scheduler_model(request):
    try:
        import pytorch_lightning as pl

        owner = importlib.import_module(f"{request.param}.sleep2vec_finetuning").Sleep2vecFinetuning
    except (AttributeError, ImportError, RuntimeError) as exc:
        if "torchvision" in str(exc):
            pytest.skip(f"Local optional vision dependencies prevent Lightning import: {exc}")
        raise

    class SchedulerModel(pl.LightningModule):
        configure_optimizers = owner.configure_optimizers
        on_validation_epoch_end = owner.on_validation_epoch_end

        def lr_scheduler_step(self, scheduler, metric):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                owner.lr_scheduler_step(self, scheduler, metric)
            else:
                super().lr_scheduler_step(scheduler, metric)

        def _log_layer_mix_weights(self, **kwargs):
            pass

        def _get_eval_model(self):
            return self.model

        def _finalize_epoch(self, stage):
            score = 1.0
            if self.improving and self.current_epoch >= 3:
                score = 0.5 if self.args.monitor_mod == "min" else 2.0
            self.log("val_score", score, on_epoch=True)

        def __init__(self, mode="min", improving=False):
            super().__init__()
            self.args = SimpleNamespace(
                lr=1.0,
                weight_decay=0.0,
                lr_scheduler="plateau",
                lr_decay_floor=0.25,
                lr_plateau_factor=0.5,
                lr_plateau_patience=0,
                monitor="val_score",
                monitor_mod=mode,
                check_val_every_n_epoch=2,
            )
            self.model = torch.nn.Linear(1, 1)
            self._finetune_param_to_group = {"weight": "encoder", "bias": "head"}
            self._finetune_lr_scales = {"encoder": 0.1, "head": 1.0}
            self.tuning_config = SimpleNamespace(groups=["encoder", "head"], preset="full")
            self.improving = improving
            self.epoch_lrs = []

        def training_step(self, batch, batch_idx):
            return self.model(batch[0]).square().mean()

        def validation_step(self, batch, batch_idx):
            pass

        def on_train_epoch_start(self):
            self.epoch_lrs.append([group["lr"] for group in self.optimizers().param_groups])

    return pl, SchedulerModel


def _fit(pl, model, epochs, checkpoint=None):
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=2,
    )
    loader = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader, ckpt_path=checkpoint)
    return trainer


def test_plateau_steps_only_on_validation_and_preserves_group_floor(scheduler_model):
    pl, model_class = scheduler_model
    model = model_class()
    trainer = _fit(pl, model, 8)
    assert [lr[1] for lr in model.epoch_lrs] == pytest.approx([1, 1, 1, 1, 0.5, 0.5, 0.25, 0.25])
    for encoder_lr, head_lr in model.epoch_lrs:
        assert encoder_lr / head_lr == pytest.approx(0.1)
    assert [group["lr"] for group in trainer.optimizers[0].param_groups] == pytest.approx([0.025, 0.25])


@pytest.mark.parametrize("mode", ["min", "max"])
def test_plateau_uses_monitor_direction(scheduler_model, mode):
    pl, model_class = scheduler_model
    model = model_class(mode=mode, improving=True)
    trainer = _fit(pl, model, 6)
    assert [lr[1] for lr in model.epoch_lrs] == pytest.approx([1] * 6)
    assert trainer.optimizers[0].param_groups[1]["lr"] == pytest.approx(0.5)


def test_plateau_checkpoint_restores_metric_history_and_lr(scheduler_model, tmp_path):
    pl, model_class = scheduler_model
    full = model_class()
    full_trainer = _fit(pl, full, 6)
    partial = model_class()
    partial_trainer = _fit(pl, partial, 4)
    checkpoint = tmp_path / "plateau.ckpt"
    partial_trainer.save_checkpoint(checkpoint)
    restored = model_class()
    restored_trainer = _fit(pl, restored, 6, str(checkpoint))
    assert restored.epoch_lrs == full.epoch_lrs[4:]
    assert restored_trainer.lr_scheduler_configs[0].scheduler.state_dict() == (
        full_trainer.lr_scheduler_configs[0].scheduler.state_dict()
    )


def test_plateau_reduces_tiny_groups_proportionally(scheduler_model):
    _, model_class = scheduler_model
    model = model_class()
    model._finetune_lr_scales["encoder"] = 1e-8
    optimizers, schedulers = model.configure_optimizers()
    scheduler = schedulers[0]["scheduler"]
    scheduler.step(1.0)
    scheduler.step(1.0)
    assert [group["lr"] for group in optimizers[0].param_groups] == pytest.approx([5e-9, 0.5], abs=1e-15)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"warmup_steps": 5}, "does not support warmup_steps"),
        ({"lr_decay_ratio": 0.2}, "only supported by WSD"),
        ({"monitor": "train_loss"}, "requires a validation monitor"),
        ({"lr_scheduler": "wsd"}, "WSD requires lr_decay_ratio"),
        ({"lr_scheduler": "decay"}, "only supported by plateau"),
    ],
)
def test_scheduler_rejects_incompatible_runtime_options(scheduler_model, overrides, message):
    _, model_class = scheduler_model
    model = model_class()
    for name, value in overrides.items():
        setattr(model.args, name, value)
    with pytest.raises(ValueError, match=message):
        model.configure_optimizers()


@pytest.mark.parametrize("save_epoch", [1, 3])
def test_plateau_validation_checkpoint_resume_matches_uninterrupted(scheduler_model, tmp_path, save_epoch):
    pl, model_class = scheduler_model
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=tmp_path,
        filename="{epoch}",
        save_top_k=-1,
        save_on_train_epoch_end=False,
    )
    full = model_class()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=6,
        logger=False,
        callbacks=[checkpoint],
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=2,
    )
    loader = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    trainer.fit(full, train_dataloaders=loader, val_dataloaders=loader)
    checkpoint_path = tmp_path / f"epoch={save_epoch}.ckpt"
    saved = torch.load(checkpoint_path, weights_only=False)
    assert saved["lr_schedulers"][0]["last_epoch"] == (save_epoch + 1) // 2
    expected_lr = 1.0 if save_epoch == 1 else 0.5
    assert saved["optimizer_states"][0]["param_groups"][1]["lr"] == pytest.approx(expected_lr)
    restored = model_class()
    resumed_trainer = _fit(pl, restored, 6, str(checkpoint_path))
    assert restored.epoch_lrs == full.epoch_lrs[save_epoch + 1 :]
    assert resumed_trainer.lr_scheduler_configs[0].scheduler.state_dict() == (
        trainer.lr_scheduler_configs[0].scheduler.state_dict()
    )


def test_standalone_validation_does_not_step_plateau(scheduler_model):
    pl, model_class = scheduler_model
    model = model_class()
    trainer = _fit(pl, model, 4)
    scheduler = trainer.lr_scheduler_configs[0].scheduler
    state = scheduler.state_dict().copy()
    loader = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    trainer.validate(model, dataloaders=loader)
    assert scheduler.state_dict() == state


def test_plateau_train_epoch_checkpoints_resume_after_validation(scheduler_model, tmp_path):
    pl, model_class = scheduler_model
    full = model_class()
    full_trainer = _fit(pl, full, 6)
    partial = model_class()
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=tmp_path,
        filename="{epoch}",
        save_top_k=-1,
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=4,
        logger=False,
        callbacks=[checkpoint],
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=2,
    )
    loader = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    trainer.fit(partial, train_dataloaders=loader, val_dataloaders=loader)
    for filename, completed_epochs, validation_count, head_lr in [
        ("epoch=2.ckpt", 3, 1, 1.0),
        ("epoch=3.ckpt", 4, 2, 0.5),
        ("last.ckpt", 4, 2, 0.5),
    ]:
        checkpoint_path = tmp_path / filename
        saved = torch.load(checkpoint_path, weights_only=False)
        assert saved["lr_schedulers"][0]["last_epoch"] == validation_count
        assert saved["optimizer_states"][0]["param_groups"][1]["lr"] == pytest.approx(head_lr)
        restored = model_class()
        resumed_trainer = _fit(pl, restored, 6, str(checkpoint_path))
        assert restored.epoch_lrs == full.epoch_lrs[completed_epochs:]
        assert resumed_trainer.lr_scheduler_configs[0].scheduler.state_dict() == (
            full_trainer.lr_scheduler_configs[0].scheduler.state_dict()
        )


@pytest.mark.parametrize("patience,expected_epochs", [(2, 6), (4, 10)])
def test_early_stopping_counts_validation_checks_and_allows_plateau_updates(scheduler_model, patience, expected_epochs):
    pl, model_class = scheduler_model
    model = model_class()
    model.args.lr_plateau_patience = 2
    early_stop = pl.callbacks.EarlyStopping(monitor="val_score", mode="min", patience=patience)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=20,
        logger=False,
        callbacks=[early_stop],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=2,
    )
    loader = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    assert len(model.epoch_lrs) == expected_epochs
    assert early_stop.stopped_epoch == expected_epochs - 1
    assert early_stop.wait_count == patience
    assert trainer.lr_scheduler_configs[0].scheduler.last_epoch == expected_epochs // 2
    if patience == 4:
        assert [lr[1] for lr in model.epoch_lrs] == pytest.approx([1.0] * 8 + [0.5] * 2)
    else:
        assert [lr[1] for lr in model.epoch_lrs] == pytest.approx([1.0] * 6)
