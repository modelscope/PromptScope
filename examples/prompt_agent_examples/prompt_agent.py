from pathlib import Path

from prompt_scope.core.utils.utils import load_yaml
from prompt_scope.core.optimizer.research_optimizers.prompt_agent_optimizer.prompt_agent import PromptAgent
from prompt_scope.core.optimizer.research_optimizers.prompt_agent_optimizer.tasks import get_task
from prompt_scope.core.llms.dashscope_llm import DashscopeLLM, DashScopeLlmName
from prompt_scope.core.models.search_algo.loading import load_search_algo
from prompt_scope.core.models.world_model.loading import load_world_model

config_path = Path(__file__).parent.joinpath("prompt_agent_cn.yml")
config_params = load_yaml(config_path)
optim_llm = DashscopeLLM(model=DashScopeLlmName.QWEN_MAX, temperature=0.1)
base_llm = DashscopeLLM(model=DashScopeLlmName.QWEN_PLUS, temperature=0.0)

task = get_task(config_params["task_name"])(**config_params)
world_model = load_world_model(
    model=config_params["world_model_name"], 
    task=task,
    base_model=base_llm,
    optim_model=optim_llm,
    **config_params)
search_algo = load_search_algo(
    algo=config_params["search_algo_name"],
    task=task,
    world_model=world_model,
    **config_params)
pagent = PromptAgent(task=task, search_algo=search_algo, **config_params)
best_prompt = pagent.run()
print(best_prompt)