from meta_icl.core.offline.demonstration_augmentation.generation_by_beam_search import BeamSearchGenerationByDiversity

if __name__ == '__main__':
    num_expand = 5
    demonstration_generation_instruction = "请根据提供的样例，给出${num_generated_examples}个类似样例，要求和现在的样例的任务类型一致。\n\n要求：\n1. 生成的语言和提供的参考样例保持一致， 即提供的参考样例是英文的，你给出的样例也应该是英文的；如果提供的参考样例是中文的，你给出的样例也应该是中文的\n2. 给出的样例尽量与参考样例属于同一个任务类型，但和参考样例有较大区别，并且是不同domain的。\n3. 和提供的参考样例保持一致输出格式，并且每个样例必须用markdown json形式单独区分。即输出形式:\n```json\n你生成的样例1\n```\n\n```json\n你生成的样例2\n```\n\n${other_requirements}\n\n参考样例：\n${demonstration}\n\n\n请给出${num_generated_examples}个类似样例:"
    demonstration_requirements = ""
    demonstration_dir = "logs/gsm8k_example/results"
    model_config = {
        "module_name": 'dashscope_generation',
        "model_name": "qwen-plus",
        "clazz": 'models.llama_index_generation_model',
        "max_tokens": 2000,
        "seed": 1234,
        "temperature": 1
    }

    diversity_generator = BeamSearchGenerationByDiversity(
        demonstration_save_dir=demonstration_dir,
        num_expand=num_expand,
        demonstration_generation_instruction=demonstration_generation_instruction,
        demonstration_requirements=demonstration_requirements,
        auto_save=True,
        expand_model_config=model_config
    )
    seed_demonstrations = [
        {
            "input": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
                     "How many clips did Natalia sell altogether in April and May?",
            "output": "To find out how many clips Natalia sold altogether in April and May, follow these steps:\n\n1. "
                      "**April Sales**: Natalia sold clips to 48 friends in April.\n\n2. **May Sales**: It's "
                      "mentioned that she sold half as many clips in May as she did in April. So, to find out how "
                      "many clips she sold in May, you divide the April sales by 2.\n\n   \\[ \\text{May Sales} = "
                      "\\frac{\\text{April Sales}}{2} = \\frac{48}{2} = 24 \\]\n\n3. **Total Sales (April + May)**: "
                      "To find the total number of clips sold in both months, add the sales from April to the sales "
                      "from May.\n\n   \\[ \\text{Total Sales} = \\text{April Sales} + \\text{May Sales} = 48 + 24 = "
                      "72 \\]\n\nSo, Natalia sold a total of 72 clips altogether in April and May.\n#### 72"
        },
        {
            "input": "A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate "
                     "its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger "
                     "over time, so each new ship has twice as many people as the last ship. How many people were on "
                     "the ship the monster ate in the first hundred years?",
            "output": "Let's denote the number of people on the first ship (100 years ago) as \\(x\\).\n\nAccording "
                      "to the problem, each subsequent ship has twice as many people as the previous one. So, "
                      "in the second hundred years, the ship had \\(2x\\) people, and in the third hundred years, "
                      "the ship had \\(4x\\) people (since \\(2 \\times 2 = 4\\), doubling each time).\n\nThe total "
                      "number of people eaten over the three hundred years is given as 847. We can set up an equation "
                      "to represent this:\n\n\\[x + 2x + 4x = 847\\]\n\nCombining like terms gives us:\n\n\\[7x = "
                      "847\\]\n\nTo find \\(x\\), divide both sides of the equation by 7:\n\n\\[x = \\frac{847}{"
                      "7}\\]\n\n\\[x = 121\\]\n\nTherefore, there were 121 people on the ship that the monster ate in "
                      "the first hundred years.\n#### 121"
        },
        {
            "input": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How "
                     "much did she earn? ",
            "output": "To calculate Weng's earnings for 50 minutes of babysitting at a rate of $12 per hour, "
                      "follow these steps:\n\n1. **Determine the hourly rate**: The rate is already given as $12 per "
                      "hour.\n\n2. **Convert babysitting time to hours**: Since the rate is in dollars per hour, "
                      "you need to convert the 50 minutes into hours. There are 60 minutes in an hour, so you divide "
                      "the number of minutes by 60.\n\n\\[ \\text{Time in hours} = \\frac{\\text{Time in minutes}}{"
                      "60} = \\frac{50}{60} \\]\n\n3. **Calculate the fraction of an hour**: \n\n\\[ \\frac{50}{60} = "
                      "\\frac{5}{6} \\text{ hours} \\]\n\n4. **Calculate earnings**: To find out how much she earned, "
                      "multiply the fraction of the hour she worked by her hourly rate.\n\n\\[ \\text{Earnings} = "
                      "\\text{Hourly rate} \\times \\text{Fraction of an hour worked} \\]\n\n\\[ \\text{Earnings} = "
                      "\\$12 \\times \\frac{5}{6} \\]\n\n5. **Perform the multiplication**:\n\n\\[ \\text{Earnings} = "
                      "\\$12 \\times \\frac{5}{6} = \\$10 \\]\n\nSo, Weng earned $10 for 50 minutes of "
                      "babysitting.\n#### 10"
        }
    ]
    diversity_generator.run(seed_demonstrations=seed_demonstrations, n=10, max_steps=2, beam_width=2)
