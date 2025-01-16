from __future__ import annotations

import json
import os
from typing import Generic, List, Dict, Any, Tuple
from loguru import logger
from pathlib import PosixPath

from prompt_scope.core.optimizer.research_optimizers.prompt_agent_optimizer.tasks.base_task import BaseTask
from prompt_scope.core.models.search_algo.base_algo import BaseSearchAlgo
from prompt_scope.core.models.search_algo.nodes import BeamNode, State, Action

class BeamSearch(BaseSearchAlgo):
    '''
        sample beam_width + 1 batches of data
        generate steps_per_gradient * beam_width new prompts
        for each prompt in beam:
            generate steps_per_gradient new prompts
        use score to evaluate all the prompts
        => actions? = list( (prompt, reward))
        preserve beam_width new prompts as state
        new state = reverse_sorted(actions)[:beam_width]
    '''

    def __init__(
            self,
            task: BaseTask,
            world_model: Generic[State, Action],

            beam_width: int = 3,
            expand_width: int =3,
            depth_limit: int = 8,

            test_all_nodes: bool =False,
            **kwargs,
    ) -> None:

        self.task = task
        self.world_model = world_model

        self.expand_width = expand_width
        self.depth_limit = depth_limit
        self.beam_width = beam_width
        self.test_all_nodes = test_all_nodes

        self.nodes: List[BeamNode] = []
        self.all_nodes: List[BeamNode] = []

        self.log_vars()

    def log_vars(self):
        logger.info('-------------------- Beam Search -----------------------')
        ignored_print_vars = ['nodes', 'all_nodes']
        vars_dict = vars(self)
        for var_name in vars_dict:
            if var_name in ignored_print_vars: continue
            var_value = vars_dict[var_name]
            logger.info(f'{var_name} : {var_value}')
        logger.info('-------------------------------------------')

    def _expand(self, node: BeamNode) -> List[BeamNode]:
        logger.info(f'------------------  expand node {node.id} ---------------------')
        while True:
            new_nodes = []
            batch = self.world_model.get_train_batch()
            children, gradient_descent_output = self.world_model.step(node, batch)
            if gradient_descent_output['acc'] == -1:
                logger.info('All correct, sample new batch.')
                continue
            for child_node in children:
                self.world_model.evaluate_node(node=child_node)  # also update threshold
            new_nodes.extend(children)
            break
        return new_nodes

    def _sort_helper(self, metric: Any) -> Any:
        if isinstance(metric, tuple):
            return metric[0]
        else:
            return metric

    def before_search(self, init_state: str) -> None:
        self.root = self.world_model.build_root(init_state)
        self.all_nodes.append(self.root)
        nodes = []
        for i in range(self.expand_width):
            new_nodes = self._expand(self.root)
            nodes.extend(new_nodes)

        nodes = sorted(nodes, key=lambda node: self._sort_helper(node.eval_metric), reverse=True)[:self.beam_width]
        self.nodes = nodes

    def search(self, i_step) -> List[BeamNode]:
        logger.info(
            f'----------------  iteration {i_step} ----------------')
        return self.nodes

    def update_nodes(self, path=List[BeamNode]) -> Tuple[List[BeamNode], List[float]]:
        nodes = []
        for node in self.nodes:
            new_nodes = self._expand(node)
            nodes.extend(new_nodes)
        nodes = sorted(nodes, key=lambda node: self._sort_helper(node.eval_metric), reverse=True)[:self.beam_width]
        self.all_nodes.extend(nodes)
        self.nodes = nodes
        return self.nodes, [x.eval_metric for x in nodes]

    def after_search(self, store_path: PosixPath) -> Tuple[List[BeamNode], Dict[str, List[List[BeamNode]] | List[BeamNode] | BeamNode]]:
        output = self.prepare_output()
        self.output_to_json(store_path=store_path, output=output)

        return self.nodes, output

    def __call__(self, init_state: str, **kwargs) -> Tuple[List[BeamNode], Dict[str, List]]:
        BeamNode.reset_id()

        self.before_search(init_state=init_state)
        for i in self.depth_limit:
            logger.info(
            f'----------------  iteration {i} ----------------')
            self.update_nodes()
            
        nodes, output = self.after_search()
        return nodes, output

    def test_and_log_node(self, node: BeamNode, eval=False, eval_type='test'):
        if eval:
            if eval_type == 'test':
                test_metric, eval_output = self.world_model.test_prompt(node.prompt)
            else:
                raise ValueError(f'eval_type {eval_type} is not supported.')
            node.test_metric = test_metric
        if node.parent is not None:
            logger.info(
                f'node {node.id}:    parent: {node.parent.id} | depth: {node.depth} | eval: {node.eval_metric} | test: {node.test_metric}\nprompt: {node.prompt}')
        else:
            logger.info(
                f'node {node.id}:    parent: N/A | depth: {node.depth} | eval: {node.eval_metric} | test: {node.test_metric}\nprompt: {node.prompt}')
        logger.info(f'---------------------')

    def test_and_log_nodes(self, nodes, eval=False):
        for node in nodes:
            self.test_and_log_node(node=node, eval=eval)

    def prepare_output(self) -> Dict[str, List[List[BeamNode]] | List[BeamNode] | BeamNode]:
        # test and log nodes
        logger.info(f'\n---------------------  test nodes ------------------------')
        self.test_and_log_nodes(nodes=self.nodes, eval=True)
        # prepare output
        paths_nodes: List[List[BeamNode]] = []

        for i, node in enumerate(self.nodes):
            path: List[BeamNode] = []
            while node.parent is not None:
                path.append(node)
                node = node.parent
            path = path[::-1]
            paths_nodes.append(path)
            logger.info(f'---------------------  path {i} ------------------------')
            self.test_and_log_nodes(path, eval=False)
        best_path = sorted(paths_nodes, key=lambda path: self._sort_helper(path[-1].eval_metric), reverse=True)[0]
        best_node = sorted(self.all_nodes, key=lambda node: self._sort_helper(node.eval_metric), reverse=True)[0]

        if len(self.world_model.test_dataloader) != 0:
            logger.info(f'---------------------  best path ------------------------')
            self.test_and_log_nodes(best_path, eval=True)

            logger.info(f'---------------------  best path node------------------------')
            self.test_and_log_node(best_path[-1], eval=False)

            logger.info(f'---------------------  best global node------------------------')
            self.test_and_log_node(best_node, eval=True)

        return dict(
            all_paths=paths_nodes,
            best_path=best_path,
            best_path_node=[best_path[-1]],
            best_global_node=[best_node]
        )

    def output_to_json(self, store_path: PosixPath, output: Dict[str, List[List[BeamNode]] | List[BeamNode] | BeamNode]) -> None:
        data_to_save = {}
        paths = []
        for path in output['all_paths']:
            paths.append([node.to_dict() for node in path])
        data_to_save['all_paths'] = paths

        for key in output:
            if key != "all_paths":
                data_to_save[key] = [node.to_dict() for node in output[key]]
        with open(os.path.join(store_path, 'data.json'), 'w') as f:
            json.dump(data_to_save, f, indent=4)
