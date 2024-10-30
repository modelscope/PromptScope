from typing import Dict, Type, Any

from prompt_scope.core.models.search_algo.base_algo import SearchAlgoName, BaseSearchAlgo
from prompt_scope.core.models.search_algo.beam_search import BeamSearch
from prompt_scope.core.models.search_algo.mcts import MCTS

_SEARCHALGO_MAP: Dict[
    SearchAlgoName, Type[BaseSearchAlgo]] = {
    SearchAlgoName.MCTS: MCTS,
    SearchAlgoName.BEAM_SEARCH: BeamSearch,
}


def load_search_algo(
    algo: SearchAlgoName,
    **kwargs: Any,
) -> BaseSearchAlgo:
    """Load the requested search algorithm specified by a string.

    Parameters
    ----------
    algo : SearchAlgoName
        The type of search algorithms to load.
    **kwargs : Any
        Additional keyword arguments to pass to the algorithm.

    Returns
    -------
    BaseSearchAlgo
        The loaded search algorithm.
    """
    if algo not in _SEARCHALGO_MAP:
        raise ValueError(
            f"Unknown search algorithm type: {algo}"
            f"\nValid types are: {list(_SEARCHALGO_MAP.keys())}"
        )
    algo_cls = _SEARCHALGO_MAP[algo]
    return algo_cls(**kwargs)