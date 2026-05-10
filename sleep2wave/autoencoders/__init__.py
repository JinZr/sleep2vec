import importlib

_MODEL_EXPORTS = {"Sleep2WaveAutoencoder", "Sleep2WaveAutoencoderOutput"}
_LOSS_EXPORTS = {"Sleep2WaveAutoencoderLoss", "compute_autoencoder_loss"}
_LIGHTNING_EXPORTS = {"Sleep2WaveAutoencoderLightning"}
_CHECKPOINT_EXPORTS = {"load_sleep2wave_autoencoder_checkpoint"}


def __getattr__(name):
    if name in _MODEL_EXPORTS:
        module = importlib.import_module("sleep2wave.autoencoders.model")
        return getattr(module, name)
    if name in _LOSS_EXPORTS:
        module = importlib.import_module("sleep2wave.autoencoders.losses")
        return getattr(module, name)
    if name in _LIGHTNING_EXPORTS:
        module = importlib.import_module("sleep2wave.autoencoders.lightning")
        return getattr(module, name)
    if name in _CHECKPOINT_EXPORTS:
        module = importlib.import_module("sleep2wave.autoencoders.checkpoints")
        return getattr(module, name)
    raise AttributeError(name)


__all__ = [
    "Sleep2WaveAutoencoder",
    "Sleep2WaveAutoencoderLightning",
    "Sleep2WaveAutoencoderLoss",
    "Sleep2WaveAutoencoderOutput",
    "compute_autoencoder_loss",
    "load_sleep2wave_autoencoder_checkpoint",
]
