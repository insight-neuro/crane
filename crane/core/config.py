from transformers import PretrainedConfig


class BrainConfig(PretrainedConfig):
    model_type = "brain_model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
