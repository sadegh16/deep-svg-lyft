from abc import ABC

import torch
import pytorch_lightning as pl
import typing as th

from raster.models.model_trajectory import ModelTrajectory
from raster.utils import KeyValue, boolify

from argparse import ArgumentParser
from pytorch_lightning.utilities.distributed import rank_zero_only


class LyftTrainerModule(pl.LightningModule, ABC):
    def __init__(
            self,
            model_config,
            model: str = 'Resnet',
            model_dict: dict = None,
            modes: int = 1,
            optimizer: str = 'Adam',
            optimizer_dict: th.Optional[dict] = None,
            lr: float = 1e-4,
            scheduler: th.Optional[str] = None,
            scheduler_dict: th.Optional[dict] = None,
            scheduler_interval: str = 'epoch',
            scheduler_frequency: int = 1,
            scheduler_monitor: th.Optional[str] = None,
            saliency_factor: float = 0.,
            saliency_intrest: str = 'simple',
            saliency_dict: th.Optional[dict] = None,
            pgd_mode: str = 'loss',
            pgd_reg_factor: float = 0.,
            pgd_iters: int = 0,
            pgd_alpha: float = 0.01,
            pgd_random_start: bool = False,
            pgd_eps_vehicles: float = 0.4,
            pgd_eps_semantics: float = 0.15625,
            track_grad: bool = False,
            test_csv_path: str = None,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)
        self.model = ModelTrajectory(model_cfg=self.hparams.model_config, modes=self.hparams.modes, future_len=30,
                                     in_channels=3)
        # optimization & scheduling
        self.lr = self.hparams.lr
        self.track_grad = self.hparams.track_grad
        self.val_hparams = 0.
        self.criterion = torch.nn.MSELoss()

    @rank_zero_only  # todo check
    def init_hparam_logs(self):
        metrics = ['loss', 'nll', 'grads/total']
        val = 1e3  # todo must not affect graphs
        metric_placeholder = {
            **{f'{m}/train': val for m in metrics}, **{f'{m}/val': val for m in metrics},
            'saliency/val': 0., 'saliency/train': 0.,
        }
        if self.logger:
            self.logger.log_hyperparams(self.hparams, metrics=metric_placeholder)

    def forward(self, inputs, targets: torch.Tensor, return_results=True,
                grad_enabled=True, return_trajectory=False):
        torch.set_grad_enabled(grad_enabled)
        res = dict()
        ##added
        model_args = [inputs[arg] for arg in self.hparams.model_config.model_args]
        entery = [*model_args, {}, True]
        pred, _ = self.model(entery)
        #         print(targets)
        #         print(pred)
        loss = self.criterion(targets, pred.reshape(targets.shape), )
        if return_trajectory:
            res['loss'] = loss
            res['pred'] = pred
        else:
            res['loss'] = loss.mean()

        if return_results:
            return res
        return res['loss']

    def on_train_batch_start(self, batch, batch_idx: int, dataloader_idx: int):
        # print(batch['valid'])
        batch['image']['commands'] = batch['image']['commands'][batch['valid']]
        if len(batch['image']['commands']) == 0:
            return -1
        batch['image']['args'] = batch['image']['args'][batch['valid']]
        batch['target_positions'] = batch['target_positions'][batch['valid']]
        return

    def step(self, batch, batch_idx, optimizer_idx=None, name='train'):
        is_val = name == 'val'
        is_test = name == 'test'
        if self.global_step == 0:
            self.init_hparam_logs()
        result = self(batch['image'], batch['target_positions'],
                      return_results=True, return_trajectory=is_test)
        # if not is_test:
        for item, value in result.items():
            self.log(f'{item}/{name}', value.mean(), on_step=not (is_val or is_test), on_epoch=is_val or is_test,
                     logger=True, sync_dist=True)
        if is_test:
            return result
        if not is_val:
            return result['loss']

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        return self.step(batch, batch_idx, optimizer_idx, name='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, name='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, name='test')

    def configure_optimizers(self):
        opt_class, opt_dict = torch.optim.Adam, {'lr': self.lr}
        if self.hparams.optimizer:
            opt_class = getattr(torch.optim, self.hparams.optimizer)
            opt_dict = self.hparams.optimizer_dict or dict()
            opt_dict['lr'] = self.lr
        opt = opt_class(self.parameters(), **opt_dict)
        if self.hparams.scheduler is None:
            return opt
        sched_class = getattr(torch.optim.lr_scheduler, self.hparams.scheduler)
        sched_dict = self.hparams.scheduler_dict or dict()
        sched = sched_class(opt, **sched_dict)
        sched_instance_dict = dict(
            scheduler=sched, interval=self.hparams.scheduler_interval, frequency=self.hparams.scheduler_frequency,
            reduce_on_plateau=isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
        )
        if self.hparams.scheduler_monitor:
            sched_instance_dict['monitor'] = self.hparams.scheduler_monitor
        return [opt], [sched_instance_dict]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model', type=str, default='Resnet', help='model architecture class to use')
        parser.add_argument('--model-dict', nargs='*', default=dict(), action=KeyValue,
                            help='additional model specific args')
        parser.add_argument('--modes', type=int, default=1, help='number of modes of model prediction')
        parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer to use')
        parser.add_argument('--optimizer-dict', nargs='*', action=KeyValue, help='additional optimizer specific args')
        parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        parser.add_argument('--scheduler', type=str, default=None, help='scheduler to use')
        parser.add_argument('--scheduler-dict', nargs='*', action=KeyValue, help='additional scheduler specific args')
        parser.add_argument('--scheduler-interval', type=str, default='epoch',
                            help='interval to call scheduler.step [step/epoch]')
        parser.add_argument('--scheduler-frequency', type=int, default=1,
                            help='frequency of calling scheduler.step on each interval')
        parser.add_argument('--scheduler-monitor', type=str, default=None,
                            help='metric to monitor for scheduling process')
        parser.add_argument('--pgd-mode', type=str, default='loss', help='pgd attack mode [loss/negative_sample]')
        parser.add_argument('--pgd-reg-factor', type=float, default=0,
                            help='whether to use adversaries as a regularization term')
        parser.add_argument('--pgd-iters', type=int, default=0, help='pgd attack number of iterations')
        parser.add_argument('--pgd-random-start', type=boolify, default=False,
                            help='whether to use a random point for adversary search during pgd attack')
        parser.add_argument('--pgd-alpha', type=float, default=1e-2, help='pgd attack alpha value')
        parser.add_argument('--pgd-eps-vehicles', type=float, default=0.4,
                            help='epsilon bound for pgd attack on vehicle layers')
        parser.add_argument('--pgd-eps-semantics', type=float, default=0.15625,
                            help='epsilon bound for pgd attack on semantic layers')
        parser.add_argument('--saliency-factor', type=float, default=0., help='saliency supervision factor')
        parser.add_argument('--saliency-intrest', type=str, default='simple',
                            help='intrest region calculation for saliency supervision')
        parser.add_argument('--saliency-dict', nargs='*', action=KeyValue,
                            help='additional saliency supervision specific args')
        parser.add_argument('--track-grad', type=boolify, default=False,
                            help='whether to log grad norms')
        return parser
