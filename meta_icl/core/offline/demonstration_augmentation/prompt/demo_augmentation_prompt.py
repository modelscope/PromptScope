Default_Instruction_4_Similar_Demonstration_Generation = """请根据提供的样例，给出${num_generated_examples}个与提供的样例相似的样例。

要求：
1. 给出的样例内容与参考样例内容要保持一致，只有用词不一致。
2. 和提供的参考样例保持一致输出格式，并且每个样例用markdown json 形式单独区分。
${other_requirements}

参考样例：
```json
${demonstration}
```

请给出${num_generated_examples}个类似样例:
"""

Default_Instruction_4_Diverse_Demonstration_Generation = """请根据提供的样例，给出${num_generated_examples}个类似样例。

要求：
1. 给出的样例尽量与参考样例属于同一个任务类型，但语言表达风格、句式等应和所提供的样例有较大差别。
2. 和提供的参考样例保持一致输出格式，并且每个样例用markdown json 形式单独区分。
${other_requirements}

参考样例：
```json
${demonstration}
```

请给出${num_generated_examples}个类似样例:
"""
