from meta_icl.core.utils.utils import revert_combined_session_2_csv
import pandas as pd

import argparse
parser = argparse.ArgumentParser(description='Agent followups')

parser.add_argument("--data_pth", type=str,
                    help="the path of the data", default=None)
parser.add_argument("--data_dir", type=str,
                    help="the dir of the data", default=None)


def convert_json_2_xlx(data_pth):
    csv_data, csv_pth = revert_combined_session_2_csv(data_pth)

    xlsx_pth = csv_pth.replace('.csv', '.xlsx')

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_pth)

    # Save the DataFrame to an XLSX file
    df.to_excel(xlsx_pth, index=False)

    return xlsx_pth

if __name__ == '__main__':
    args = parser.parse_args()
    data_pth = args.data_pth
    data_dir = args.data_dir

    if data_pth is not None:
        convert_json_2_xlx(data_pth)
    if data_dir is not None:
        import os
        file_list = os.listdir(data_dir)
        for file_name in file_list:
            if file_name.split('.')[-1] == "json":
                convert_json_2_xlx(os.path.join(data_dir, file_name))



