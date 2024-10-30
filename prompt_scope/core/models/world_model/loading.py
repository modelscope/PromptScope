from typing import Any

from prompt_scope.core.models.world_model.base_world_model import BaseWorldModel, WorldModelName
from prompt_scope.core.models.world_model.beam_world_model import BeamSearchWorldModel


_WORLDMODEL_MAP = {
    WorldModelName.BASE_WORLD_MODEL: BaseWorldModel,
    WorldModelName.BEAM_SEARCH_WORLD_MODEL: BeamSearchWorldModel,
}


def load_world_model(
    model: WorldModelName,
    **kwargs: Any,
) -> Any:
    """Load the requested world model specified by a string.

    Parameters
    ----------
    model : WorldModelName
        The type of world model to load.
    **kwargs : Any
        Additional keyword arguments to pass to the algorithm.

    Returns
    -------
    Generic[State, Action]
        The loaded world model.
    """
    if model not in _WORLDMODEL_MAP:
        raise ValueError(
            f"Unknown world model type: {model}"
            f"\nValid types are: {list(_WORLDMODEL_MAP.keys())}"
        )
    model_cls = _WORLDMODEL_MAP[model]
    return model_cls(**kwargs)