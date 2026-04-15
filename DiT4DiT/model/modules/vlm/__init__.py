


def get_backbone_model(config):

    training_mode = config.framework.cosmos25.training.lower()

    # Cosmos2.5 diffusion transformer feature extractor (video -> hidden)

    if training_mode == "joint":
        from .Cosmos25 import _Cosmos25_Interface
        return _Cosmos25_Interface(config)

    else:
        raise NotImplementedError(f"Backbone model is not implemented")



