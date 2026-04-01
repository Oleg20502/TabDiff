import glob
import json
import os
import pickle
import random
import shutil

import numpy as np
from tabdiff.metrics import TabMetrics
from tabdiff.modules.main_modules import Model
from tabdiff.builders.model_builders import (
    build_denoiser_backbone,
    build_recognition_model,
    ensure_denoiser_config_inplace,
)
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.trainer import Trainer
import src
import torch

from torch.utils.data import DataLoader
import argparse
import warnings

from tabdiff.experiment_logger import build_experiment_logger

from copy import deepcopy

from utils_train import TabDiffDataset

warnings.filterwarnings('ignore')


def _resolve_tabdiff_config_path(curr_dir, config_arg):
    if not config_arg:
        return os.path.join(curr_dir, 'configs', 'tabdiff_configs.toml')
    if os.path.isabs(config_arg):
        return config_arg
    return os.path.join(curr_dir, 'configs', config_arg)


def _config_to_json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__!r} is not JSON serializable")


def main(args):
    device = args.device

    ## Disable scientific numerical format
    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)

    ## Get data info
    dataname = args.dataname
    data_dir = f'data/{dataname}'
    info_path = f'data/{dataname}/info.json'
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    ## Set up flags
    is_dcr = 'dcr' in dataname

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_toml_path = _resolve_tabdiff_config_path(
        curr_dir, getattr(args, 'config', None)
    )
    toml_cfg = src.load_config(config_toml_path)
    use_var_for_exp_name = bool(toml_cfg.get('variational', {}).get('use_variational', False))

    ## Set experiment name (suffix matches TOML used for default ckpt discovery in test)
    exp_name = args.exp_name
    if args.exp_name is None:
        exp_name = 'non_learnable_schedule' if args.non_learnable_schedule else 'learnable_schedule'
    exp_name += '_variational' if use_var_for_exp_name else ''
    exp_name += '_y_only' if args.y_only else ''

    print(f"{args.mode.capitalize()} Mode is Enabled")
    num_samples_to_generate = args.num_samples_to_generate
    ckpt_path = None
    if args.mode == 'train':
        print("NEW training is started")
        raw_config = toml_cfg
    elif args.mode == 'test':
        ckpt_path = args.ckpt_path
        if ckpt_path is None:
            ckpt_parent_path = f"{curr_dir}/ckpt/{dataname}/{exp_name}"
            ckpt_path_arr = glob.glob(f"{ckpt_parent_path}/best_ema_model*")
            assert ckpt_path_arr, f"Cannot not infer ckpt_path from {ckpt_parent_path}, please make sure that you first train a model before testing!"
            ckpt_path = ckpt_path_arr[0]
        config_pkl_path = os.path.join(os.path.dirname(ckpt_path), 'config.pkl')
        if os.path.exists(config_pkl_path):
            with open(config_pkl_path, 'rb') as f:
                raw_config = pickle.load(f)
                print(f"Found cached config at {config_pkl_path}")
        else:
            print(f"No config.pkl next to checkpoint; using TOML at {config_toml_path}")
            raw_config = toml_cfg

    
    ## Creat model_save and result paths
    model_save_path, result_save_path = None, None
    if args.mode == 'train':
        model_save_path = 'debug/ckpt' if args.debug else f'{curr_dir}/ckpt/{dataname}/{exp_name}'
        result_save_path = model_save_path.replace('ckpt', 'result')  #i.e., f'{curr_dir}/results/{dataname}/{exp_name}'
    elif args.mode == 'test':
        if args.report:
            result_save_path = f"eval/report_runs/{exp_name}/{dataname}"
        else:
            result_save_path = os.path.dirname(ckpt_path).replace('ckpt', 'result')    # infer the exp_name from the ckpt_name
    raw_config['model_save_path'] = model_save_path
    raw_config['result_save_path'] = result_save_path
    if model_save_path is not None:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
    if result_save_path is not None:
        if not os.path.exists(result_save_path):
            os.makedirs(result_save_path)

    if os.path.isfile(config_toml_path):
        if args.mode == 'train' and model_save_path:
            _run_toml_dst = os.path.join(model_save_path, 'run_config.toml')
            shutil.copy2(config_toml_path, _run_toml_dst)
            print(f"Saved TOML config for this run to {_run_toml_dst}")
        elif args.mode == 'test' and result_save_path:
            _run_toml_dst = os.path.join(result_save_path, 'run_config.toml')
            shutil.copy2(config_toml_path, _run_toml_dst)
            print(f"Saved TOML config for this run to {_run_toml_dst}")
    
    ## Make everything determinstic if needed
    raw_config['deterministic'] = args.deterministic
    if args.deterministic:
        print("DETERMINISTIC MODE is enabled!!!")
        ## Set global random seeds
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        ## Ensure deterministic CUDA operations
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8'
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    ## Set debug mode parameters
    if args.debug:  # fast eval for DEBUG mode
        raw_config['train']['main']['check_val_every'] = 2
        raw_config['diffusion_params']['num_timesteps'] = 4
        raw_config['train']['main']['batch_size'] = 4096
        raw_config['sample']['batch_size'] = 10000

    ## Load training data
    batch_size = raw_config['train']['main']['batch_size']

    train_data = TabDiffDataset(dataname, data_dir, info, y_only=args.y_only, isTrain=True, dequant_dist=raw_config['data']['dequant_dist'], int_dequant_factor=raw_config['data']['int_dequant_factor'])
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )
    d_numerical, categories = train_data.d_numerical, train_data.categories
    
    val_data = TabDiffDataset(dataname, data_dir, info, y_only=args.y_only, isTrain=False, dequant_dist=raw_config['data']['dequant_dist'], int_dequant_factor=raw_config['data']['int_dequant_factor'])

    ## Load Metrics
    real_data_path = f'synthetic/{dataname}/real.csv'
    test_data_path = f'synthetic/{dataname}/test.csv'
    val_data_path = f'synthetic/{dataname}/val.csv'
    if not os.path.exists(val_data_path):
        print(f"{args.dataname} does not have its validation set. During MLE evaluation, a validation set will be splitted from the training set!")
        val_data_path = None
    if args.mode == 'train':
        metric_list = ["density"]
    else:
        if is_dcr:
            metric_list = ["dcr"]
        else:
            metric_list = [
                "density", 
                "mle", 
                "c2st",
            ]
    metrics = TabMetrics(real_data_path, test_data_path, val_data_path, info, device, metric_list=metric_list)
    
    ## Load the module and models
    raw_config['unimodmlp_params']['d_numerical'] = d_numerical
    raw_config['unimodmlp_params']['categories'] = (categories+1).tolist()  # add one for the mask category

    var_cfg = raw_config.get('variational', {})
    use_variational = bool(var_cfg.get('use_variational', False))

    # Propagate latent_dim to backbone config before the backbone is built
    if use_variational:
        raw_config['unimodmlp_params']['latent_dim'] = var_cfg['latent_dim']
    if args.y_only:
        raw_config['unimodmlp_params']['use_mlp'] = False     # drop the mlp when training the unconditional model
        raw_config['unimodmlp_params']['dim_t'] = 128   #reduce the size of the mlp
        main_model_path = args.ckpt_path
        if main_model_path is None:
            main_model_parent_path = f"{curr_dir}/ckpt/{dataname}/{exp_name.replace('_y_only', '')}"
            main_model_path_arr = glob.glob(f"{main_model_parent_path}/best_ema_model*")
            assert main_model_path_arr, f"Cannot not infer the main model's ckpt_path from {main_model_parent_path}, please make sure that you first train a main model before training the y_only model!"
            main_model_path = main_model_path_arr[0]
        
        main_model_configs = pickle.load(open(os.path.join(os.path.dirname(main_model_path), 'config.pkl'), 'rb'))
        # if learnable schedule is enabled in the main model,
        # we need to infer noise params of the target column from the main model ckpt
        # and train the y_only model with those params
        if main_model_configs['diffusion_params']['scheduler'] == "power_mean_per_column":
            from tabdiff.models.noise_schedule import PowerMeanNoise_PerColumn, LogLinearNoise_PerColumn
            if info['task_type'] == 'regression':
                noise_schedule = PowerMeanNoise_PerColumn(
                    num_numerical=main_model_configs['unimodmlp_params']['d_numerical'], 
                    **main_model_configs['diffusion_params']['noise_schedule_params']
                )
                 # the target col is placed at the first position
                raw_config['diffusion_params']['noise_schedule_params']['rho'] = noise_schedule.rho()[0].item()
            else:
                noise_schedule = LogLinearNoise_PerColumn(
                    num_categories=len(main_model_configs['unimodmlp_params']['categories']), 
                    **main_model_configs['diffusion_params']['noise_schedule_params']
                )
                # the target col is placed at the first position
                raw_config['diffusion_params']['noise_schedule_params']['k'] = noise_schedule.k()[0].item()

    ensure_denoiser_config_inplace(raw_config)
    latent_d = int(var_cfg.get("latent_dim", 0)) if use_variational else 0
    backbone = build_denoiser_backbone(
        raw_config,
        d_numerical=d_numerical,
        categories=(categories + 1).tolist(),
        latent_dim=latent_d,
    )
    model = Model(backbone, **raw_config['diffusion_params']['edm_params'])
    model.to(device)
    
    ## Create and load y_only_model for imputation
    y_only_model = None
    if args.impute:
        y_only_model_path = args.y_only_model_path
        if y_only_model_path is None:
            y_only_model_parent_path = f"{curr_dir}/ckpt/{dataname}/{exp_name}_y_only"
            y_only_model_path_arr = glob.glob(f"{y_only_model_parent_path}/best_ema_model*")
            assert y_only_model_path_arr, f"Cannot not infer y_only model's ckpt_path from {y_only_model_parent_path}, please make sure that you first train a y_only model before testing imputation!"
            y_only_model_path = y_only_model_path_arr[0]
        y_only_model_config_path = os.path.join(os.path.dirname(y_only_model_path), 'config.pkl')
        with open(y_only_model_config_path, 'rb') as f:
                y_only_model_config = pickle.load(f)
        ensure_denoiser_config_inplace(y_only_model_config)
        _yv = y_only_model_config.get("variational", {})
        y_only_latent = (
            int(_yv.get("latent_dim", 0)) if _yv.get("use_variational", False) else 0
        )
        y_only_bb = build_denoiser_backbone(
            y_only_model_config,
            d_numerical=d_numerical,
            categories=(categories + 1).tolist(),
            latent_dim=y_only_latent,
        )
        y_only_model = Model(
            y_only_bb, **y_only_model_config['diffusion_params']['edm_params']
        )
        y_only_model.to(device)
        # load weights
        state_dicts = torch.load(y_only_model_path, map_location=device)
        y_only_model.load_state_dict(state_dicts['denoise_fn'])

    if not args.y_only and not args.non_learnable_schedule:
        raw_config['diffusion_params']['scheduler'] = 'power_mean_per_column'
        raw_config['diffusion_params']['cat_scheduler'] = 'log_linear_per_column'

    ## Build optional recognition model (variational model)
    recognition_model = None
    if use_variational:
        recognition_model = build_recognition_model(
            var_cfg,
            num_numerical_features=d_numerical,
            num_classes_per_column=categories.tolist(),
            latent_dim=var_cfg['latent_dim'],
            posterior_inputs=var_cfg.get('posterior_inputs', 'x0'),
        )
        recognition_model.to(device)
    
    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=y_only_model,
        **raw_config['diffusion_params'],
        device=device,
        recognition_model=recognition_model,
        latent_dim=var_cfg.get('latent_dim', 0) if use_variational else 0,
        latent_policy=var_cfg.get('latent_policy', 'consistency') if use_variational else 'consistency',
        kl_weight=var_cfg.get('kl_weight', 1.0) if use_variational else 1.0,
    )

    def _param_counts(mod):
        if mod is None:
            return 0, 0
        params = list(mod.parameters())
        trainable = sum(p.numel() for p in params if p.requires_grad)
        total = sum(p.numel() for p in params)
        return trainable, total

    dn_tr, dn_tot = _param_counts(model)
    print(f"Denoiser: trainable: {dn_tr:,}  total: {dn_tot:,}")
    if recognition_model is not None:
        rec_tr, rec_tot = _param_counts(recognition_model)
        print(f"Recognition: trainable: {rec_tr:,}  total: {rec_tot:,}")
    else:
        print("Recognition: trainable: 0  total: 0  (variational disabled)")
    diff_tr, diff_tot = _param_counts(diffusion)
    print(
        f"Diffusion module (schedules, y_only if any): trainable: {diff_tr:,}  total: {diff_tot:,}"
    )
    diffusion.to(device)
    diffusion.train()

    ## Print the configs
    printed_configs = json.dumps(raw_config, default=_config_to_json_default, indent=4)
    print(f"The config of the current run is : \n {printed_configs}")
    
    ## Experiment logging (wandb, tensorboard, or none) — only from [train.main].logger in tabdiff_configs.toml
    project_name = f"tabdiff_{dataname}"
    raw_config['project_name'] = project_name
    train_main = dict(raw_config['train']['main'])
    log_backend = train_main.pop('logger', 'wandb')
    plot_density = train_main.pop('plot_density', True)
    if not isinstance(plot_density, bool):
        raise ValueError(
            f"Invalid [train.main].plot_density {plot_density!r}; use true or false in tabdiff_configs.toml."
        )
    if log_backend not in ('wandb', 'tensorboard', 'none'):
        raise ValueError(
            f"Invalid [train.main].logger {log_backend!r} in tabdiff_configs.toml; use 'wandb', 'tensorboard', or 'none'."
        )
    if args.debug:
        log_backend = 'none'
    raw_config['logger'] = log_backend

    tb_log_dir = None
    if log_backend == 'tensorboard':
        base = raw_config.get('model_save_path') or raw_config.get('result_save_path')
        if base is None:
            base = os.path.join(curr_dir, 'runs', dataname, exp_name)
            os.makedirs(base, exist_ok=True)
        tb_log_dir = os.path.join(base, 'tensorboard')
    logger = build_experiment_logger(
        log_backend,
        project_name=project_name,
        run_name=exp_name,
        config=raw_config if log_backend == 'wandb' else None,
        tensorboard_log_dir=tb_log_dir,
    )
    sample_batch_size = raw_config['sample']['batch_size']
    trainer = Trainer(
        diffusion,
        train_loader,
        train_data,
        val_data,
        metrics,
        logger,
        **train_main,
        sample_batch_size=sample_batch_size,
        num_samples_to_generate=num_samples_to_generate,
        model_save_path=raw_config['model_save_path'],
        result_save_path=raw_config['result_save_path'],
        device=device,
        ckpt_path=ckpt_path,
        y_only=args.y_only,
        kl_weight=var_cfg.get('kl_weight', 1.0) if use_variational else 0.0,
        kl_warmup_steps=var_cfg.get('kl_warmup_steps', 5000) if use_variational else 0,
        plot_density=plot_density,
    )
    try:
        if args.mode == 'test':
            if args.report:
                if  is_dcr:
                    trainer.report_test_dcr(args.num_runs)
                else:
                    trainer.report_test(args.num_runs)
            elif args.impute:
                imputed_sample_save_dir = f"impute/{dataname}/{exp_name}"
                trainer.test_impute(
                    args.trial_start, args.trial_size,
                    args.resample_rounds,
                    args.impute_condition,
                    imputed_sample_save_dir,
                    args.w_num,
                    args.w_cat,
                )
            else:
                trainer.test()
        else:
            ## Save config
            config_save_path = raw_config['model_save_path']
            pkl_path = os.path.join(config_save_path, 'config.pkl')
            json_path = os.path.join(config_save_path, 'config.json')
            with open(pkl_path, 'wb') as f:
                pickle.dump(raw_config, f)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(raw_config, f, indent=4, default=_config_to_json_default)
            print(f"Saved config to {pkl_path} and {json_path}")
            trainer.run_loop()
    finally:
        logger.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabDiff')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='TOML filename under tabdiff/configs/ (e.g. tabdiff_configs_variational.toml) or absolute path.',
    )

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'