"""Share encoder Parameter objects while retaining domain-local interfaces."""
from .model import shared_key


def tie_shared(destination, source):
    source_params = dict(source.named_parameters())
    destination_params = dict(destination.named_parameters())
    keys = {key for key in source_params if shared_key(key)}
    if keys != {key for key in destination_params if shared_key(key)}:
        raise ValueError('Backbones have different shared parameter names')
    for key in keys:
        if source_params[key].shape != destination_params[key].shape:
            raise ValueError(f'Incompatible shared parameter: {key}')
    for key in keys:
        parent, _, leaf = key.rpartition('.')
        module = destination.get_submodule(parent) if parent else destination
        setattr(module, leaf, source_params[key])


def unique_parameters(models):
    return list({id(p): p for model in models for p in model.parameters()
                 if p.requires_grad}.values())
