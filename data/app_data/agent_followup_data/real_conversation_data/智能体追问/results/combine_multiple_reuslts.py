from meta_icl.core.utils.utils import revert_combined_session_2_csv
import pandas as pd

import argparse
parser = argparse.ArgumentParser(description='Agent followups')

parser.add_argument("--data_dir", type=str,
                    help="the path of the data", default=None)
parser.add_argument("--in_key", type=str,
                    help="the dir of the data", default=[])
parser.add_argument("--not_in_key", type=str,
                    help="the dir of the data", default=[])


def convert_json_2_xlx(data_pth):
    csv_data, csv_pth = revert_combined_session_2_csv(data_pth)

    xlsx_pth = csv_pth.replace('.csv', '.xlsx')

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_pth)

    # Save the DataFrame to an XLSX file
    df.to_excel(xlsx_pth, index=False)

    return xlsx_pth

def check_condition(file_name, in_key, not_in_key):
    if isinstance(in_key, str):
        in_key = [in_key]

    if isinstance(not_in_key, str):
        not_in_key = [not_in_key]

    indicator = True
    for key in in_key:
        if key not in file_name:
            indicator = False
            break
    for key in not_in_key:
        if key in file_name:
            indicator = False
            break

    return indicator




if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.data_dir
    in_key = args.in_key
    out_key = args.not_in_key
    file_to_combine = []
    if data_dir is not None:
        import os
        file_list = os.listdir(data_dir)
        for file_name in file_list:
            if file_name.split('.')[-1] == "json":
                if check_condition(file_name, in_key, out_key):
                    file_to_combine.append(os.path.join(data_dir, file_name))
    print(file_to_combine)

    combined_data = {}
    results_key = ["followups", "cost"]
    combined_data_sav_pth = os.path.join(data_dir, f"combined_{in_key}")
    for single_pth in file_to_combine:
        csv_data, csv_pth = revert_combined_session_2_csv(single_pth)
        for key in csv_data.keys():
            if key not in results_key and key not in combined_data.keys():
                combined_data[key] = csv_data[key]
            if key in combined_data.keys():
                assert len(combined_data[key]) == len(csv_data[key])
            if key in results_key:
                new_key_name = key + "_" + single_pth.split('/')[-1].split('_followups_2024')[0]
                combined_data[new_key_name] = csv_data[key]

    pd_combined_data = pd.DataFrame(combined_data)
    pd_combined_data.to_csv(combined_data_sav_pth + ".csv", index=False)
    pd_combined_data.to_excel(combined_data_sav_pth + ".xlsx", index=False)







