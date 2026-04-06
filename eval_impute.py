import argparse
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from tabdiff.config import load_training_manifest
from tabdiff.utils.io import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate imputation outputs from a run directory.")
    parser.add_argument("--run-dir", required=True, help="Training run directory.")
    parser.add_argument("--name", default="impute", help="Imputation output name under run_dir/imputation/.")
    args = parser.parse_args()

    manifest = load_training_manifest(args.run_dir)
    dataset_name = manifest["run"]["dataset"]
    dataset_info = load_json(f"data/{dataset_name}/info.json")

    real_data = pd.read_csv(f"data/{dataset_name}/test.csv")
    target_col = real_data.columns[dataset_info["target_col_idx"][0]]
    impute_dir = f"{args.run_dir}/imputation/{args.name}"
    sample_paths = sorted(glob.glob(f"{impute_dir}/*.csv"))
    if not sample_paths:
        raise FileNotFoundError(f"No imputation samples found in {impute_dir}.")

    task_type = dataset_info["task_type"]
    encoder = OneHotEncoder()

    if task_type == "binclass":
        real_target = real_data[target_col].to_numpy().reshape(-1, 1)
        real_y = encoder.fit_transform(real_target).toarray()
        syn_y = []
        for syn_path in sample_paths:
            syn_data = pd.read_csv(syn_path)
            target = syn_data[target_col].to_numpy().reshape(-1, 1)
            syn_y.append(encoder.transform(target).toarray())
        syn_y_prob = np.stack(syn_y).mean(0)
        auc = roc_auc_score(real_y, syn_y_prob, average="micro")
        print("AUC:", round(auc * 100, 3))
    else:
        y_test = np.log(np.clip(real_data[target_col].to_numpy(), 1, 20000))
        syn_y = []
        for syn_path in sample_paths:
            syn_data = pd.read_csv(syn_path)
            syn_y.append(np.log(np.clip(syn_data[target_col].to_numpy(), 1, 20000)))
        pred = np.stack(syn_y).mean(0)
        rmse = root_mean_squared_error(y_test, pred)
        print("RMSE:", round(rmse, 4))


if __name__ == "__main__":
    main()
    
