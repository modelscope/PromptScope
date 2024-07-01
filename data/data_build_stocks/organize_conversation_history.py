from meta_icl.core.utils import load_csv, sav_json
from meta_icl.core.utils import get_current_date, combine_session
import pandas as pd
import os




if __name__ == '__main__':

    csv_pth = "data/data_build_stocks/others/0604.csv"
    json_sav_dir = "data/data_build_stocks/others/"
    group_by_filed = "session_id"

    combine_session(csv_pth, json_sav_dir, group_by_filed, selection_filed=None)
