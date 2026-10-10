import torch


@torch.no_grad()
def encode(model, samples, config, device, patches=False, project=False):
    """Use the retained preprocessing; evaluation never applies spectral masking."""
    from torchmdnet.data import SpectraPreprocessedDataset

    settings = dict(config, spec_augment=False, peak_mask=False)
    processed = SpectraPreprocessedDataset(
        samples, settings, is_train=False, peak_mask=False
    )
    spectra = [
        torch.stack([processed[i].spectra[m] for i in range(len(samples))]).to(device)
        for m in range(3)
    ]
    return encode_spectra(model, spectra, patches=patches, project=project)


@torch.no_grad()
def encode_spectra(model, spectra, patches=False, project=False):
    encoder = model.representation_spec_model
    masks = [
        torch.zeros(len(spectra[0]), n, dtype=torch.bool, device=spectra[0].device)
        for n in encoder.patch_nums
    ]
    captured = []
    handle = (
        encoder.pool.register_forward_pre_hook(
            lambda module, args: captured.append(args[0])
        )
        if patches
        else None
    )
    try:
        pooled = encoder(spectra, masks, len(spectra[0]))[1].unsqueeze(1)
    finally:
        if handle is not None:
            handle.remove()
    if patches:
        if len(captured) != 1:
            raise RuntimeError("Expected one encoder pooling call")
        return torch.cat([pooled, captured[0]], dim=1)
    return model.spec_projector(pooled) if project else pooled


def fuse_spectra(samples, weights):
    mixed = samples[0].clone()
    for m, name in enumerate(("uv", "ir", "raman")):
        values = [getattr(s, name).float().clamp_min(0) for s in samples]
        setattr(mixed, name, sum(float(weights[m, k]) * values[k] for k in range(3)))
    mixed.smiles = ".".join(str(s.smiles) for s in samples)
    return mixed


def ir_edit(low, high, start_cm, end_cm):
    mixed = low.clone()
    weight = crossfade_weights(start_cm, end_cm)
    left, right = low.ir.reshape(-1).float(), high.ir.reshape(-1).float()
    if left.numel() != 3501 or right.numel() != 3501:
        raise ValueError("IR editing requires the original 500–4000 cm-1 grid")
    mixed.ir = (weight * left + (1 - weight) * right).reshape_as(low.ir)
    return mixed


def crossfade_weights(start_cm: int, end_cm: int) -> torch.Tensor:
    if not 500 <= start_cm < end_cm <= 4000:
        raise ValueError("Cross-fade bounds must satisfy 500 <= start < end <= 4000.")
    grid = torch.arange(500, 4001, dtype=torch.float32)
    weights = torch.ones_like(grid)
    weights[grid >= end_cm] = 0.0
    middle = (grid > start_cm) & (grid < end_cm)
    weights[middle] = (end_cm - grid[middle]) / float(end_cm - start_cm)
    return weights
