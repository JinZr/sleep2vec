from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest


def _load_sleep2vec_adaptation_module():
    module_name = "_sleep2vec_adaptation_schedule_test"
    stubbed_modules = {
        "pytorch_lightning": ModuleType("pytorch_lightning"),
        "torch": ModuleType("torch"),
        "data.channel_selection": ModuleType("data.channel_selection"),
        "sleep2vec.checkpoints": ModuleType("sleep2vec.checkpoints"),
        "sleep2vec.config": ModuleType("sleep2vec.config"),
        "sleep2vec.sleep2vec_modelling": ModuleType("sleep2vec.sleep2vec_modelling"),
    }

    stubbed_modules["pytorch_lightning"].Callback = object
    stubbed_modules["data.channel_selection"].build_all_pairs = lambda channel_names: list(channel_names)
    stubbed_modules["sleep2vec.checkpoints"].load_pretrain_init_weights = lambda *args, **kwargs: None
    stubbed_modules["sleep2vec.config"].AdaptConfig = object
    stubbed_modules["sleep2vec.sleep2vec_modelling"].Sleep2vecPretraining = object

    originals = {name: sys.modules.get(name) for name in stubbed_modules}
    originals[module_name] = sys.modules.get(module_name)

    try:
        for name, module in stubbed_modules.items():
            sys.modules[name] = module

        spec = importlib.util.spec_from_file_location(
            module_name,
            Path(__file__).resolve().parents[2] / "sleep2vec/sleep2vec_adaptation.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load sleep2vec.sleep2vec_adaptation for testing.")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class AdaptPairScheduleCallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_sleep2vec_adaptation_module()

    def test_pair_schedule_reaches_final_stage_by_last_epoch(self) -> None:
        schedule = [
            SimpleNamespace(until=0.25, new_pair_ratio=1.0),
            SimpleNamespace(until=0.50, new_pair_ratio=0.7),
            SimpleNamespace(until=0.75, new_pair_ratio=0.5),
            SimpleNamespace(until=1.0, new_pair_ratio=0.0),
        ]
        cases = [
            (0, 4, 1.0),
            (1, 4, 0.7),
            (2, 4, 0.5),
            (3, 4, 0.0),
        ]

        for current_epoch, max_epochs, expected_ratio in cases:
            with self.subTest(current_epoch=current_epoch, max_epochs=max_epochs):
                sampler = SimpleNamespace(
                    pairs=[("eeg", "ecg"), ("eeg", "ppg"), ("ecg", "ppg")],
                    pair_probs=None,
                )
                sampler.set_pair_probs = lambda pair_probs, sampler=sampler: setattr(sampler, "pair_probs", pair_probs)
                trainer = SimpleNamespace(
                    current_epoch=current_epoch,
                    max_epochs=max_epochs,
                    train_dataloader=SimpleNamespace(batch_sampler=sampler),
                )
                callback = self.module.AdaptPairScheduleCallback(
                    new_channels=["ppg"],
                    pair_schedule=schedule,
                )

                callback.on_train_epoch_start(trainer, pl_module=None)

                expected_probs = self.module.build_new_modality_pair_probs(
                    sampler.pairs,
                    new_channels=["ppg"],
                    new_pair_ratio=expected_ratio,
                )
                self.assertEqual(set(sampler.pair_probs), set(expected_probs))
                for pair, expected_prob in expected_probs.items():
                    self.assertAlmostEqual(sampler.pair_probs[pair], expected_prob)


if __name__ == "__main__":
    unittest.main()
