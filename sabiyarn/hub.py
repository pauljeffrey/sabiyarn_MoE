"""Loading and saving SabiYarn checkpoints with the model code that ships in this repo."""

from __future__ import annotations


def register_local_model_code() -> None:
    """Make save_pretrained() write modeling.py, configuration.py and config.json's auto_map.

    Models loaded through trust_remote_code carry this registration automatically; the local
    classes (model.code: local) do not, so without it a saved checkpoint is just weights + a
    config with no auto_map. Pushed to the Hub, that overwrites config.json and breaks
    `trust_remote_code=True` loading for everyone using the repo. With it, every checkpoint ships
    the exact code that trained it (sparse dispatch, flex attention) and pushing one updates the
    Hub's code and config together. Used by training/ and rl/.
    """
    from sabiyarn.model.configuration import GPTJXMoEConfig
    from sabiyarn.model.modeling import GPTJXMoEForCausalLM

    GPTJXMoEConfig.register_for_auto_class()
    GPTJXMoEForCausalLM.register_for_auto_class("AutoModelForCausalLM")
