from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def run_finetune(monkeypatch, tmp_path):
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger

    from sleep2vec2 import finetune

    class SyntheticModel(pl.LightningModule):
        def __init__(self, emit_metric):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.emit_metric = emit_metric
            self.training_epochs = []
            self.validation_epochs = []
            self.initial_state = None

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=0.01)

        def on_train_start(self):
            optimizer = self.trainer.optimizers[0]
            self.initial_state = {
                "global_step": self.global_step,
                "weight": self.weight.detach().clone(),
                "optimizer": {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in optimizer.state.get(self.weight, {}).items()
                },
            }

        def training_step(self, batch, batch_idx):
            self.training_epochs.append(self.current_epoch)
            return (self.weight - batch[0]).square().mean()

        def validation_step(self, batch, batch_idx):
            pass

        def on_validation_epoch_end(self):
            if not self.trainer.sanity_checking:
                self.validation_epochs.append(self.current_epoch)
            if self.emit_metric:
                self.log("val_cohen_kappa", float(self.current_epoch + 1))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(finetune, "persist_run_config_and_args", lambda *args: None)
    monkeypatch.setattr(finetune, "save_training_run_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(finetune, "_finish_wandb_run", lambda *args: None)
    monkeypatch.setattr(finetune, "WandbLogger", lambda **kwargs: CSVLogger(tmp_path / "logs", name=kwargs["name"]))
    monkeypatch.setattr(finetune, "DDPStrategy", lambda **kwargs: "auto")
    loader = DataLoader(TensorDataset(torch.zeros(5, 1)), batch_size=1)
    monkeypatch.setattr(finetune, "prepare_dataloader", lambda args: (loader, loader, None))

    def run(version, epochs, checkpoint="", emit_metric=True, legacy_early_stopping=False):
        model = SyntheticModel(emit_metric)
        trainers = []

        def early_stopping(**kwargs):
            if legacy_early_stopping:
                kwargs.pop("check_on_train_epoch_end", None)
            return pl.callbacks.EarlyStopping(**kwargs)

        def cpu_trainer(**kwargs):
            kwargs.update(
                accelerator="cpu",
                devices=1,
                enable_progress_bar=False,
                enable_model_summary=False,
                num_sanity_val_steps=0,
                log_every_n_steps=1,
            )
            trainer = pl.Trainer(**kwargs)
            trainers.append(trainer)
            return trainer

        monkeypatch.setattr(finetune, "pl", SimpleNamespace(Trainer=cpu_trainer))
        monkeypatch.setattr(finetune, "EarlyStopping", early_stopping)
        monkeypatch.setattr(finetune, "Sleep2vecFinetuning", lambda *args, **kwargs: model)
        args = SimpleNamespace(
            version=version,
            monitor="val_cohen_kappa",
            monitor_mod="max",
            patience=20,
            ckpt_every_n_epochs=1,
            devices=[0],
            epochs=epochs,
            gradient_clip_val=0.0,
            accumulate_grad_batches=2,
            precision="32-true",
            check_val_every_n_epoch=1,
            print_diagnostics=False,
            ckpt_path=str(checkpoint),
            test_after_fit=False,
            test_all_checkpoints_after_fit=False,
            label_name="stage5",
        )
        bundle = SimpleNamespace(model=SimpleNamespace(channels=[]), averaging=None, finetune=None)
        finetune.supervised(args, bundle)
        return model, trainers[0]

    return run


def test_validation_checkpoint_resume_preserves_state_and_full_epoch_budget(run_finetune):
    _, source_trainer = run_finetune("source", 2, legacy_early_stopping=True)
    best = next(callback for callback in source_trainer.checkpoint_callbacks if callback.monitor == "val_cohen_kappa")
    checkpoint_path = Path(best.best_model_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 1
    assert checkpoint["global_step"] == 6
    progress = checkpoint["loops"]["fit_loop"]
    assert progress["epoch_loop.batch_progress"]["is_last_batch"]
    assert progress["epoch_loop.val_loop.batch_progress"]["is_last_batch"]
    assert progress["epoch_progress"]["current"]["completed"] == 1

    model, trainer = run_finetune("resumed", 12, checkpoint_path)

    assert model.initial_state["global_step"] == 6
    assert torch.equal(model.initial_state["weight"], checkpoint["state_dict"]["weight"])
    optimizer_state = checkpoint["optimizer_states"][0]["state"][0]
    for key, value in optimizer_state.items():
        assert torch.equal(model.initial_state["optimizer"][key], value)
    assert Counter(model.training_epochs) == {epoch: 5 for epoch in range(2, 12)}
    assert model.validation_epochs == list(range(2, 12))
    assert trainer.global_step == 6 + 10 * 3
    assert trainer.current_epoch == 12
    assert not trainer.should_stop


def test_early_stopping_still_rejects_missing_validation_metric(run_finetune):
    with pytest.raises(RuntimeError, match="Early stopping conditioned on metric `val_cohen_kappa`"):
        run_finetune("missing-metric", 1, emit_metric=False)
