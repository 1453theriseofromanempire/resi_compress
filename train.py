import torch
import torch.nn as nn
import torch.distributed as dist

from omegaconf import OmegaConf
import argparse
import sys, math

from torchvision import datasets, transforms
from torchvision.datasets import ImageNet

from resi.models.resi_model import ResiModel

sys.path.append("/root/home/codes/resi_compress/src/rq-vae-transformer-main")
from rqvae.utils.setup import setup
from rqvae.trainers import create_trainer
from rqvae.optimizer import create_optimizer, create_scheduler
import rqvae.utils.dist as dist_utils


# def get_parser(**parser_kwargs):
#     def str2bool(v):
#         if isinstance(v, bool):
#             return v
#         if v.lower() in ("yes", "true", "t", "y", "1"):
#             return True
#         elif v.lower() in ("no", "false", "f", "n", "0"):
#             return False
#         else:
#             raise argparse.ArgumentTypeError("Boolean value expected.")

#     parser = argparse.ArgumentParser(**parser_kwargs)
#     parser.add_argument(
#         "-n",
#         "--name",
#         type=str,
#         const=True,
#         default="",
#         nargs="?",
#         help="postfix for logdir",
#     )
#     parser.add_argument(
#         "-r",
#         "--resume",
#         type=str,
#         const=True,
#         default="",
#         nargs="?",
#         help="resume from logdir or checkpoint in logdir",
#     )
#     parser.add_argument(
#         "-b",
#         "--base",
#         nargs="*",
#         metavar="base_config.yaml",
#         help="paths to base configs. Loaded from left-to-right. "
#         "Parameters can be overwritten or added with command-line options of the form `--key value`.",
#         default=list(),
#     )
#     parser.add_argument(
#         "-t",
#         "--train",
#         type=str2bool,
#         const=True,
#         default=False,
#         nargs="?",
#         help="train",
#     )
#     parser.add_argument(
#         "--no-test",
#         type=str2bool,
#         const=True,
#         default=False,
#         nargs="?",
#         help="disable test",
#     )
#     parser.add_argument("-p", "--project", help="name of new or path to existing project")
#     parser.add_argument(
#         "-d",
#         "--debug",
#         type=str2bool,
#         nargs="?",
#         const=True,
#         default=False,
#         help="enable post-mortem debugging",
#     )
#     parser.add_argument(
#         "-s",
#         "--seed",
#         type=int,
#         default=23,
#         help="seed for seed_everything",
#     )
#     parser.add_argument(
#         "-f",
#         "--postfix",
#         type=str,
#         default="",
#         help="post-postfix for default name",
#     )
#     parser.add_argument(
#         "-l",
#         "--config",
#         type=str,
#         help="config path"
#     )

#     return parser

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model-config', type=str, default='./configs/c10-igpt.yaml')
parser.add_argument('-r', '--result-path', type=str, default='./results.tmp')
parser.add_argument('-l', '--load-path', type=str, default='')
parser.add_argument('-t', '--test-batch-size', type=int, default=200)
parser.add_argument('-e', '--test-epoch', type=int, default=-1)
parser.add_argument('-p', '--postfix', type=str, default='')
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
parser.add_argument('--node_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--timeout', type=int, default=86400, help='time limit (s) to wait for other nodes in DDP')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--resume', action='store_true')

args, extra_args = parser.parse_known_args()

def read_cfg(path):
    config = OmegaConf.load(path)
    return config

def create_dataset(image_folder_path, is_eval=False, logger=None):
    # Define transformations for training and validation datasets
    # transforms_trn = create_transforms(config.dataset, split='train', is_eval=is_eval)
    transforms_ = [
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
    transforms_val = transforms.Compose(transforms_)

    dataset = ImageNet(root_path=image_folder_path, transform=transforms_val)

    return dataset # Returning the same dataset for both training and validation

def train(model, train_dataset):
    model.train()

def main():
    config, logger, writer = setup(args, extra_args)
    distenv = config.runtime.distenv
    # config = read_cfg(parser.config)

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda', distenv.local_rank)
    torch.cuda.set_device(device)

    model = ResiModel(config.ddconfig, config.fusionconfig)
    model_ema = None
    model.to(device)

    trainer = create_trainer(config)

    dataset_trn, dataset_val = create_dataset(config, is_eval=args.eval, logger=logger)

    train_epochs = config.experiment.epochs
    steps_per_epoch =  math.ceil(len(dataset_trn) / config.experiment.batch_size)
    epoch_st = 0
    if not args.eval:
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(
            optimizer, config.optimizer.warmup, steps_per_epoch,
            config.experiment.epochs,
        )
        
    disc_state_dict = None
    if not args.load_path == '':
        ckpt = torch.load(args.load_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        disc_state_dict = ckpt.get('discriminator', None)

    model = dist_utils.dataparallel_and_sync(distenv, model)
    trainer = trainer(model, model_ema, dataset_trn, dataset_val, config, writer,
                    device, distenv, disc_state_dict=disc_state_dict)

    trainer.run_epoch(optimizer, scheduler, epoch_st)

    dist.barrier()

    if distenv.master:
        writer.close()  # may prevent from a file stable error in brain cloud..

if __name__ == "__main__":

    main()