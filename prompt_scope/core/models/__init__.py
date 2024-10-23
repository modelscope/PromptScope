from .base_model import MODEL_REGISTRY
from .beam_world_model import BeamSearchWorldModel
from .world_model import WorldModel

# from .generation_model import GenerationModel, AioGenerationModel

world_models = {
    "mcts": WorldModel,
    "beam_search": BeamSearchWorldModel,
}
MODEL_REGISTRY.batch_register(world_models)

# __all__ = [
#     "WorldModel",
#     "BeamSearchWorldModel",
#     "GenerationModel",
#     "AioGenerationModel",
#     "MODEL_REGISTRY",
# ]
