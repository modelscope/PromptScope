from meta_icl.core.utils.sys_prompt_utils import check_dir, sav_json
from meta_icl.core.utils.utils import get_current_date
import json, os
from typing import List


# from meta_icl.core.utils.demontration_utils import generate_similar_demonstration, demonstration_var_score


class GenerationByBeamSearch:
    def __init__(self,
                 demonstration_save_dir,
                 demonstration_expand,
                 demonstration_var_score,
                 auto_save=True):
        """
        :param demonstration_var_score
        :param generate_similar_demonstration: func
        :param demonstration_save_dir: func
        """
        check_dir(demonstration_save_dir)
        self.demonstration_save_dir = demonstration_save_dir
        self.expand_fn = demonstration_expand
        self.score_fn = demonstration_var_score
        self.all_states = []
        self.auto_save = auto_save

    def _renew_all_state(self):
        self.all_states = []

    def _update_demonstration_file_name(self, prefix, date, model_name):
        self.sav_file_name = "{}_{}_{}.json".format(prefix, date, model_name)

    def _add_demonstrations(self, demonstration: List or dict or str):
        if isinstance(demonstration, List):
            if isinstance(demonstration[0], str):
                self.all_states.extend([json.loads(item) for item in demonstration])
            else:
                self.all_states.extend(demonstration)

        if isinstance(demonstration, dict):
            self.all_states.append(demonstration)

        if isinstance(demonstration, str):
            self.all_states.append(json.loads(demonstration))
        if self.auto_save:
            self._sav_demonstration()

    def _sav_demonstration(self):
        json_file_path = os.path.join(self.demonstration_save_dir, self.sav_file_name)
        print("save self.all_state to pth: {}".format(json_file_path))
        sav_json(data=self.all_states, json_file_path=json_file_path)

    def beam_search(self, max_steps, beam_width, expand_config):
        """
            Performs beam search.

            :param initial_state: List The initial state to begin search from.
            :param max_steps: The maximum number of steps (or iterations) to perform.
            :param beam_width: The number of paths to explore at each step.
            :param expand_fn: A function that takes a state and returns a list of possible next states.
            :param score_fn: A function that takes a state and returns a score. Higher scores are better.
            :param expand_fn_config: the config for the expand_fn, a function that takes a state and returns a list of possible next states.
            :return: The best state found according to the scoring function.
            """

        initial_state = expand_config["initial_demonstration"]
        print("initial state: {}".format(initial_state))
        print("type of init: {}".format(type(initial_state)))

        self._renew_all_state()
        self._update_demonstration_file_name(prefix="demonstration",
                                             date=get_current_date(),
                                             model_name=expand_config["model_name"])
        self._add_demonstrations(initial_state)

        # Initialize the beam with the initial state.
        beam = [[initial_state, self.score_fn(initial_state)]]
        print("beam: {}".format(beam))

        for step in range(max_steps):
            # Expand states in the current beam.
            print("step: {}".format(step))
            candidates = []
            for state, score in beam:
                next_states = self.expand_fn(state, expand_config)
                print("next_states: \n{}".format(next_states))
                for next_state in next_states:
                    self._add_demonstrations(next_states[-1])
                    next_score = self.score_fn(next_state)
                    print("next_score: {}".format(next_score))
                    candidates.append([next_state, next_score])

            # Select the top `beam_width` states for the next iteration.
            print(candidates)
            print("length of candidates: {}".format(len(candidates)))
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_width]

        # Return the state with the highest score after the final iteration.
        best_state, best_score = max(beam, key=lambda x: x[1])
        return best_state, self.all_states
