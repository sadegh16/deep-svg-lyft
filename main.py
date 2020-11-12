import pytorch_lightning as pl
from pytorch_lightning import loggers
from l5kit.configs import load_config_data
from raster.lyft import LyftTrainerModule, LyftDataModule
from pathlib import Path
import argparse
import torch
from raster.utils import boolify
import importlib

parser = argparse.ArgumentParser(description='Manage running job')
parser.add_argument('--seed', type=int, default=313, help='random seed to use')
parser.add_argument('--config', type=str, required=True, help='config yaml path')
parser.add_argument("--config-model", type=str, required=True)
parser.add_argument('--log-lr', type=boolify, default=True, help='learning rate log interval')
parser.add_argument('--log-gpu-stats', type=boolify, default=False, help='whether to monitor gpu stats')
parser.add_argument('--log-root', type=str, default='./lightning_logs', help='experiments logs root')
parser.add_argument('--name', type=str, default=None, help='experiments logs root')
parser.add_argument('--transfer', type=str, default=None, help='initial weights to transfer on')
parser = LyftTrainerModule.add_model_specific_args(parser)
parser = LyftDataModule.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)

if __name__ == '__main__':
    args = parser.parse_args()
    # initializing various parts
    pl.seed_everything(args.seed)

    # initializing training
    callbacks = []
    if args.log_lr and args.scheduler:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval=args.scheduler_interval))
    if args.log_gpu_stats and torch.cuda.is_available():
        callbacks.append(pl.callbacks.GPUStatsMonitor())
    Path(args.log_root).mkdir(parents=True, exist_ok=True)  # make sure log-root path exists
    logger = loggers.TensorBoardLogger(save_dir=args.log_root, name='default' if not args.name else args.name,
                                       default_hp_metric=False, log_graph=False)
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='loss/val', save_last=True, verbose=True, mode='min')
    trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint, callbacks=callbacks, logger=logger)
    config = load_config_data(args.config)
    model_config = importlib.import_module(args.config_model).Config()
    model_name, experiment_name = args.config_model.split(".")[-2:]
    args_dict = vars(args)
    args_dict['config'] = config
    args_dict['model_config'] = model_config
    training_procedure = LyftTrainerModule(**args_dict)
    if args.transfer is not None:
        training_procedure.load_state_dict(torch.load(args.transfer)['state_dict'])
        print(args.transfer, 'loaded as initial weights')
    args_dict['config'] = training_procedure.hparams.config
    training_procedure.datamodule = LyftDataModule(**args_dict)
    trainer.fit(training_procedure)
