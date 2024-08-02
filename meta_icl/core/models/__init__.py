from meta_icl.core.models.beam_world_model import BeamSearchWorldModel
from meta_icl.core.models.world_model import WorldModel
from meta_icl.core.models.base_model import MODEL_REGISTRY

world_models = {
    "mcts": WorldModel,
    "beam_search": BeamSearchWorldModel,
}
MODEL_REGISTRY.batch_register(world_models)
