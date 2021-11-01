from modeling.encoder import MeLT
from test_tube import HyperOptArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
import torch.nn as nn
import pytorch_lightning as pl
import sys
import os
import random
from pytorch_lightning import Callback
import optuna
from optuna.integration import PyTorchLightningPruningCallback

class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

def get_args(model):
    
    parent_parser = HyperOptArgumentParser(strategy='random_search', add_help=False)

    root_dir = os.getcwd()

    parent_parser.add_argument('--mode', type=str, default='default', choices=('default', 'test', 'param_search'), help='Toggle train/test/search modes')
    parent_parser.add_argument('--save-path', type=str, default='/data/mmatero/melt/models', help='directory to save models')
    parent_parser.add_argument('--checkpoint', type=str, default='2', help="file of saved checkpoint")
    parent_parser.add_argument('--version', type=str, default='1492', help="version number of checkpoint")
    parent_parser.add_argument('--gpus', type=str, default='0,1,2', help='GPU IDs as CSV')
    parent_parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'), help="distributed processing protocol")
    parent_parser.add_argument('--use_16bit', dest='use_16bit', action='store_true', help='use 16bit precision')
    parent_parser.add_argument('--num_trials', default=5, type=int, help='number of trials for param search')

    # helpful debugging
    parent_parser.add_argument('--fast_dev_run', dest='fast_dev_run', action='store_true', help='debug a full train/test/val loop')
    parent_parser.add_argument('--track_grad_norm', dest='track_grad_norm', action='store_true', help='toggles grad norm tracking')

    parser = model.add_model_specific_args(parent_parser, root_dir)
    return parser

def run_model(hparams, gpus=None):
    # Default train/test loop from scratch
    if hparams.mode == 'default':
        model = MeLT(hparams)
    
    # Load in pretrained model
    if hparams.mode == 'test':
        version_path = hparams.save_path + '/lightning_logs/version_' + hparams.version
        checkpoint_path = version_path + '/checkpoints/epoch=' + hparams.checkpoint + '.ckpt'
    
        meta_path = version_path + '/meta_tags.csv'
        print(f'Loading model from {checkpoint_path}')
        model = MeLT.load_from_checkpoint(checkpoint_path, tags_csv=meta_path)

        if hparams.max_msg_seq_dev != model.max_msg_dev:
            model.max_msg_dev = hparams.max_msg_seq_dev
            model.hparams.max_msg_seq_dev = hparams.max_msg_seq_dev
    
    if hparams.seed is not None:
        random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        torch.backends.cudnn.deterministic = True
 
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00005,
        patience=3,
        verbose=False,
        mode='min'
    )

    # Set model setup config
    trainer = pl.Trainer(default_save_path=hparams.save_path,
                         gpus=len(gpus.split(',')) if gpus else hparams.gpus,
                         distributed_backend=hparams.distributed_backend,
                         use_amp=hparams.use_16bit,
                         max_nb_epochs=hparams.epochs,
                         #gradient_clip_val=.25,
                         track_grad_norm=(2 if hparams.track_grad_norm else -1),
                         early_stop_callback=early_stop_callback)
   
    # only train model in default, run test for default and test configs
    #if hparams.mode == 'default':
    if hparams.mode == 'default':
        trainer.fit(model)
        trainer.test()
    elif hparams.mode == 'test':
        trainer.test(model)
    
def objective(trial, gpus=None):
    # unique file names
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(hparams.save_path, 'trial_{}'.format(trial.number), '{epoch}'), monitor='val_loss')

    # save metrics from each validation step
    metrics_callback = MetricsCallback()
    trainer = pl.Trainer(logger=False,
                         checkpoint_callback=checkpoint_callback,
                         max_nb_epochs=hparams.epochs,
                         gpus=len(gpus.split(',')) if gpus else hparams.gpus,
                         callbacks=[metrics_callback],
                         early_stop_callback=PyTorchLightningPruningCallback(trial, monitor='val_loss'))
    # setup model
    model = MeLT(hparams, trial)
    trainer.fit(model)

    return metrics_callback.metrics[-1]['val_loss']

if __name__ == '__main__':
    print('Starting...')
    hparams = get_args(MeLT)
    hparams = hparams.parse_args()
    
    if hparams.mode == 'param_search':
        pruner = optuna.pruners.MedianPruner() 
        study = optuna.create_study(direction='minimize', pruner=pruner)
        study.optimize(objective, n_trials=hparams.num_trials)

        print('Num Trials Completed: {}'.format(len(study.trials)))
        trial = study.best_trial
        print('Best Trial: {}'.format(trial.value))

        for key,value in trial.params.items():
            print('{}:{}'.format(key,value))
    elif hparams.mode in ['test', 'default']:
        run_model(hparams)

