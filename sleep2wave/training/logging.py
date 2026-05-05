SLEEP2WAVE_DIFFUSION_PROJECT = "sleep2wave-diffusion"


def build_diffusion_run_name(version_name: str, *, phase: int, context_epochs: int) -> str:
    if not version_name:
        raise ValueError("version_name must be non-empty.")
    return f"sleep2wave-diffusion-{version_name}-phase{phase}-ctx{context_epochs}"


__all__ = ["SLEEP2WAVE_DIFFUSION_PROJECT", "build_diffusion_run_name"]
