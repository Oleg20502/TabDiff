# TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation

<p align="center">
  <a href="https://github.com/MinkaiXu/TabDiff/blob/main/LICENSE">
    <img alt="MIT License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a>
  <a href="https://openreview.net/forum?id=swvURjrt8z">
    <img alt="Openreview" src="https://img.shields.io/badge/review-OpenReview-blue">
  </a>
  <a href="https://arxiv.org/abs/2410.20626">
    <img alt="Paper URL" src="https://img.shields.io/badge/cs.LG-2410.20626-B31B1B.svg">
  </a>
</p>

<div align="center">
  <img src="images/tabdiff_demo.gif" alt="Model Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <p><em>Figure 1: Visualing the generative process of TabDiff. A high-quality version of this video can be found at <a href="images/tabdiff_demo.mp4" download>tabdiff_demo.mp4</a></em></p>
</div>

This repository provides the official implementation of TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation (ICLR 2025).

## Latest Update
- [2025.04]：The categorical-heavy dataset **[Diabetes](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)** evaluated in the paper has now been released!
- [2025.02]：Our code is finally released! We have released part of the tested datasets. The rest will be released soon!

## Introduction

<div align="center">
  <img src="images/tabdiff_flowchart.jpg" alt="Model Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <p><em>Figure 2: The high-level schema of TabDiff</a></em></p>
</div>
TabDiff is a unified diffusion framework designed to model all muti-modal distributions of tabular data in a single model. Its key innovations include:  

1) Framing the joint diffusion process in continuous time,
2) A feature-wised learnable diffusion process that offsets the heterogeneity across different feature distributions,
3) Classifier-free guidance conditional generation for missing column value imputation. 

The schema of TabDiff is presented in the figure above. For more details, please refer to [our paper](https://arxiv.org/abs/2410.20626).


## Environment Setup

Create the main environment with [tabdiff.yaml](tabdiff.yaml). This environment will be used for all tasks except for the evaluation of additional data fidelity metrics (i.e., $\alpha$-precision and $\beta$-recall scores)

```
conda env create -f tabdiff.yaml
```

Create another environment with [synthcity.yaml](synthcity.yaml) to evaluate additional data fidelity metrics

```
conda env create -f synthcity.yaml
```

## Datasets Preparation

### Using the datasets experimented in the paper

Download raw datasets:

```
python download_dataset.py
```

Process datasets:

```
python process_dataset.py
```

### Using your own dataset

First, create a directory for your dataset in [./data](./data):
```
cd data
mkdir <NAME_OF_YOUR_DATASET>
```

Compile your raw tabular data in .csv format. **The first row should be the header** indicating the name of each column, and the remaining rows are records. After finishing these steps, place you data's csv file in the directory you just created and name it as <NAME_OF_YOUR_DATASET>.csv. 

Then, create <NAME_OF_YOUR_DATASET>.json in [./data/Info](./data/Info). Write this file with the metadata of your dataset, covering the following information:
```
{
    "name": "<NAME_OF_YOUR_DATASET>",
    "task_type": "[NAME_OF_TASK]", # binclass or regression
    "header": "infer",
    "column_names": null,
    "num_col_idx": [LIST],  # list of indices of numerical columns
    "cat_col_idx": [LIST],  # list of indices of categorical columns
    "target_col_idx": [list], # list of indices of the target columns (for MLE)
    "file_type": "csv",
    "data_path": "data/<NAME_OF_YOUR_DATASET>/<NAME_OF_YOUR_DATASET>.csv"
    "test_path": null,
}
```

### Important Notes When Creating the Info File
- The MLE evaluation and the imputation task (see later sections for details) assume that one column of your data is the regression or classification target. To enable these tasks, you will need to specify `target_col_idx`. If you don't need to evalute MLE, you can comment out the following line: https://github.com/MinkaiXu/TabDiff/blob/0c4fc3bbfa19046d36c5dce64628df52d5c73d15/tabdiff/main.py#L152
- The fields `target_col_idx`, `num_col_idx` and `cat_col_idx` must be multually exclusive—no column should appear in more than one of these lists. 
- Set the task_type to "regression" if the target column is numerical, or "binclass" if it is categorical.

Finally, run the following command to process your dataset:
```
python process_dataset.py --dataname <NAME_OF_YOUR_DATASET>
```

## Training TabDiff

Training and testing now use separate entrypoints and YAML configs.

Train a model with:

```bash
python train.py --config tabdiff/configs/train/unimod_mlp/base.yaml --device cuda:0
```

The bundled training configs are organized by backbone:

- `tabdiff/configs/train/unimod_mlp/`
- `tabdiff/configs/train/transformer_encoder/`
- `tabdiff/configs/train/mlp/`

Variational examples are included as separate YAML files, for example:

```bash
python train.py --config tabdiff/configs/train/unimod_mlp/variational.yaml --device cuda:0
```

All hyperparameters now live in YAML. CLI is intentionally limited to runtime arguments such as config path, device, and optional run-name override.

Each training run writes everything under a single directory:

```text
runs/<dataset>/<run_name>/
```

This directory contains configs, checkpoints, logs, training-time evaluation outputs, test outputs, report outputs, and imputation outputs.

## Sampling and Evaluating TabDiff (Density, MLE, C2ST)

After training, run testing from the saved run directory:

```bash
python test.py --run-dir runs/adult/unimod_mlp_base --config tabdiff/configs/test/sample.yaml --device cuda:0
```

To run report mode:

```bash
python test.py --run-dir runs/adult/unimod_mlp_base --config tabdiff/configs/test/report.yaml --device cuda:0
```

Report outputs are saved under:

```text
runs/<dataset>/<run_name>/report/<report_name>/
```

## Evaluating on Additional Fidelity Metrics ($\alpha$-precision and $\beta$-recall scores)

First generate report samples with the report test config. Then switch to the `synthcity` environment:

```bash
conda activate synthcity
```

Run:

```bash
python eval/eval_quality.py --run-dir runs/adult/unimod_mlp_base --name report
```

The extra fidelity metrics are appended inside the same report directory.

## Evaluating Data Privacy (DCR score)

For DCR, preprocess and train the `_dcr` dataset variant, then run the report test config against that run directory. Since outputs are run-local now, DCR results also stay under the same run directory.

## Missing Value Imputation with Classifier-free Guidance (CFG)

Our current experiments only include imputing the target column. The imputation path is implemented in `sample_impute()` in [unified_ctime_diffusion.py](./tabdiff/models/unified_ctime_diffusion.py).

### Training Guidance Model

Use a y-only training config, for example:

```bash
python train.py --config tabdiff/configs/train/unimod_mlp/y_only.yaml --device cuda:0
```

Set `model.y_only_source_run_dir` in that config to the main model run you want to pair with.

### Sampling Imputed Tables

Set `test.imputation.guidance_run_dir` in `tabdiff/configs/test/impute.yaml`, then run:

```bash
python test.py --run-dir runs/adult/unimod_mlp_base --config tabdiff/configs/test/impute.yaml --device cuda:0
```

Imputation outputs are saved under:

```text
runs/<dataset>/<run_name>/imputation/<name>/
```

### Evaluating Imputation

```bash
python eval_impute.py --run-dir runs/adult/unimod_mlp_base --name impute
```

## License

This work is licensed undeer the MIT License.

## Acknowledgement
This repo is built upon the previous work TabSyn's [[codebase]](https://github.com/amazon-science/tabsyn). Many thanks to Hengrui!

## Citation
Please consider citing our work if you find it helpful in your research!
```
@inproceedings{
shi2025tabdiff,
title={TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation},
author={Juntong Shi and Minkai Xu and Harper Hua and Hengrui Zhang and Stefano Ermon and Jure Leskovec},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=swvURjrt8z}
}
```
## Contact
If you encounter any problem, please file an issue on this GitHub repo.

If you have any question regarding the paper, please contact Minkai at [minkai@stanford.edu](minkai@stanford.edu) or Juntong at [shisteve@usc.edu](shisteve@usc.edu).
