import random
from typing import List, Dict, Any

from prompt_scope.core.utils.cluster_util import emb_cluster, emb_cluster2
from prompt_scope.core.embeddings.base import BaseEmbedding


def choose_data(
        candidate_badcases: List[Dict[str, Any]],
        strategy: str,
        candidate_goodcases: List[Dict[str, Any]]=None,
        embedding: BaseEmbedding=None):
    chosen_goodcases = []
    chosen_badcases = []
    if strategy == 'all':
        chosen_goodcases = candidate_goodcases
        chosen_badcases = candidate_badcases
    elif strategy == 'rand20':
        random.shuffle(candidate_goodcases)
        random.shuffle(chosen_badcases)
        chosen_goodcases = candidate_goodcases[:20]
        chosen_badcases = candidate_badcases[:20]
    elif strategy == 'rand10':
        random.shuffle(candidate_goodcases)
        random.shuffle(candidate_badcases)
        chosen_goodcases = candidate_goodcases[:10]
        chosen_badcases = candidate_badcases[:10]
    elif strategy == 'rand5':
        random.shuffle(candidate_goodcases)
        random.shuffle(candidate_badcases)
        chosen_goodcases = candidate_goodcases[:5]
        chosen_badcases = candidate_badcases[:5]
    elif strategy == 'rand5_bad':
        random.shuffle(candidate_goodcases)
        random.shuffle(candidate_badcases)
        chosen_goodcases = []
        chosen_badcases = candidate_badcases[:5]
    elif strategy == 'query_cluster5c4':
        chosen_goodcases = emb_cluster(candidate_goodcases, n_clusters=5, num_per_cluster=4, embedding=embedding)
        chosen_badcases = emb_cluster(candidate_badcases,  n_clusters=5, num_per_cluster=4, embedding=embedding)
    elif strategy == 'query_cluster10c2':
        chosen_goodcases = emb_cluster(candidate_goodcases,  n_clusters=10, num_per_cluster=2, embedding=embedding)
        chosen_badcases = emb_cluster(candidate_badcases,  n_clusters=10, num_per_cluster=2, embedding=embedding)
    elif strategy == 'all10c2':
        chosen_badcases, chosen_goodcases = emb_cluster2(candidate_badcases, candidate_goodcases, 0, n_cluster=10, cluster_num=2)
    elif strategy == 'all5c2':
        chosen_badcases, chosen_goodcases = emb_cluster2(candidate_badcases, candidate_goodcases, 0, n_cluster=5, cluster_num=2)
    elif strategy == 'all10c1':
        chosen_badcases, chosen_goodcases = emb_cluster2(candidate_badcases, candidate_goodcases, 0, n_cluster=10, cluster_num=1)
    elif strategy == 'all10c3':
        chosen_badcases, chosen_goodcases = emb_cluster2(candidate_badcases, candidate_goodcases, 0, n_cluster=10, cluster_num=3)
    elif strategy == 'all10c4':
        chosen_badcases, chosen_goodcases = emb_cluster2(candidate_badcases, candidate_goodcases, 0, n_cluster=10, cluster_num=4)
    elif strategy == 'label_cluster':
        chosen_badcases = emb_cluster(candidate_badcases, 2, n_cluster=10, cluster_num=2)
        chosen_goodcases = emb_cluster(candidate_goodcases, 2, n_cluster=10, cluster_num=2)
    return chosen_goodcases, chosen_badcases

