import pandas as pd
from typing import Dict, List

def save_eval_details(eval_details: List[Dict[str, str]], save_path: str):
    result = {}
    for d in eval_details:
        for key, value in d.items():
            if key not in result:
                result[key] = []
            result[key].append(value)

    pd.DataFrame(result).to_excel(save_path)