import glob
import os
import os.path as osp
import shutil

import torch
from mmcv.runner import save_checkpoint
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.hook import HOOKS, Hook


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


@HOOKS.register_module()
class TransferWeight(Hook):
    
    def __init__(self, every_n_inters=1):
        self.every_n_inters=every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())


@HOOKS.register_module()
class TrainableParamPrefixHook(Hook):
    """Keep only selected parameter-prefix modules trainable during training."""

    def __init__(self, trainable_param_prefixes):
        self.trainable_param_prefixes = tuple(trainable_param_prefixes)
        self.trainable_module_prefixes = tuple(
            prefix[:-1] if prefix.endswith('.') else prefix
            for prefix in self.trainable_param_prefixes)

    def _apply(self, runner):
        model = _unwrap_model(runner.model)
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(self.trainable_param_prefixes)

        model.eval()
        for name, module in model.named_modules():
            if name.startswith(self.trainable_module_prefixes):
                module.train()

    def before_train_epoch(self, runner):
        self._apply(runner)

    def before_train_iter(self, runner):
        self._apply(runner)


@HOOKS.register_module()
class BestLatestPlanningCheckpointHook(Hook):
    """Save only latest.pth and best.pth (lowest planning reg sum)."""

    def __init__(self,
                 interval=1,
                 metric_keys=('planning.loss_diffusion_reg_0',
                              'planning.loss_diffusion_reg_1'),
                 save_optimizer=True,
                 out_dir=None):
        self.interval = interval
        self.metric_keys = metric_keys
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.best_score = float('inf')

    def before_run(self, runner):
        if self.out_dir is None:
            self.out_dir = runner.work_dir
        best_path = osp.join(self.out_dir, 'best.pth')
        if osp.isfile(best_path):
            try:
                ckpt = torch.load(best_path, map_location='cpu')
                meta = ckpt.get('meta', {})
                self.best_score = float(meta.get('best_planning_reg', self.best_score))
            except Exception:
                runner.logger.warning('Failed to read best.pth meta; reset best score.')

    def _epoch_planning_reg(self, runner):
        metrics = runner.log_buffer.output
        score = 0.0
        found = False
        for key in self.metric_keys:
            if key in metrics:
                score += float(metrics[key])
                found = True
        return score, found

    def _save_file(self, runner, filepath):
        optimizer = runner.optimizer if self.save_optimizer else None
        meta = dict(best_planning_reg=self.best_score)
        if runner.meta is not None:
            meta.update(runner.meta)
        meta.update(epoch=runner.epoch + 1, iter=runner.iter)
        save_checkpoint(runner.model, filepath, optimizer=optimizer, meta=meta)

    def _remove_epoch_checkpoints(self, runner):
        for ckpt_path in glob.glob(osp.join(self.out_dir, 'epoch_*.pth')):
            try:
                os.remove(ckpt_path)
            except OSError as exc:
                runner.logger.warning(f'Failed to remove {ckpt_path}: {exc}')

    @master_only
    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return

        latest_path = osp.join(self.out_dir, 'latest.pth')
        self._save_file(runner, latest_path)
        runner.logger.info(f'Saved latest checkpoint: {latest_path}')

        score, found = self._epoch_planning_reg(runner)
        best_path = osp.join(self.out_dir, 'best.pth')
        if found and score < self.best_score:
            self.best_score = score
            shutil.copy2(latest_path, best_path)
            ckpt = torch.load(best_path, map_location='cpu')
            if ckpt.get('meta') is None:
                ckpt['meta'] = {}
            ckpt['meta']['best_planning_reg'] = self.best_score
            torch.save(ckpt, best_path)
            runner.logger.info(
                f'Updated best checkpoint: {best_path} (planning_reg={score:.4f})')
        elif not osp.isfile(best_path):
            shutil.copy2(latest_path, best_path)
            runner.logger.info(f'Initialized best checkpoint: {best_path}')

        self._remove_epoch_checkpoints(runner)

