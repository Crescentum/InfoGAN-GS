

import argparse
import torch
from trainer_chair import InfoGANTrainer, TrainerConfig, VALID_MODES


def parse_args():
    p = argparse.ArgumentParser(description='InfoGAN training')
    p.add_argument('--mode',       type=str, default='vanilla',
                   choices=list(VALID_MODES))
    p.add_argument('--dataset',    type=str, default='mnist',
                   choices=['mnist', 'svhn', 'celeba', 'chairs'])
    p.add_argument('--epochs',     type=int, default=50)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--data_dir',   type=str, default='./data')
    p.add_argument('--log_dir',    type=str, default='./logs')
    p.add_argument('--ckpt_dir',   type=str, default='./checkpoints')
    p.add_argument('--lr_d',       type=float, default=2e-4)
    p.add_argument('--lr_g',       type=float, default=1e-3)
    p.add_argument('--lambda_gp',  type=float, default=10.0)
    p.add_argument('--lambda_disc',type=float, default=1.0)
    p.add_argument('--lambda_cont',type=float, default=0.1)
    p.add_argument('--infonce_temp',type=float, default=0.1)
    p.add_argument('--resume',     type=str, default=None,
                   help='path to .pt checkpoint to resume from')
    p.add_argument('--start_epoch',type=int, default=None,
                   help='override start epoch when resuming (default: checkpoint epoch + 1)')
    p.add_argument('--gumbel_temp', type=float, default=1.0,
               help='Gumbel-Softmax temperature (improvement: use 0.5)')
    return p.parse_args()


def main():
    args = parse_args()

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        resumed_dataset = ckpt.get('dataset', args.dataset)
        resumed_mode    = ckpt.get('mode', args.mode)
        resumed_epoch   = ckpt.get('epoch', -1)
        
        if args.dataset != 'mnist' and args.dataset != resumed_dataset:
            print(f"[Warning] Overriding --dataset {args.dataset} -> {resumed_dataset}")
        if args.mode != 'vanilla' and args.mode != resumed_mode:
            print(f"[Warning] Overriding --mode {args.mode} -> {resumed_mode}")
        
        args.dataset = resumed_dataset
        args.mode    = resumed_mode
        print(f"[Resume] Checkpoint: dataset={resumed_dataset}, mode={resumed_mode}, epoch={resumed_epoch}")

    cfg = TrainerConfig(
        mode            = args.mode,
        dataset         = args.dataset,
        max_epochs      = args.epochs,
        batch_size      = args.batch_size,
        data_dir        = args.data_dir,
        log_dir         = args.log_dir,
        checkpoint_dir  = args.ckpt_dir,
        lr_d            = args.lr_d,
        lr_g            = args.lr_g,
        lambda_gp       = args.lambda_gp,
        lambda_disc     = args.lambda_disc,
        lambda_cont     = args.lambda_cont,
        infonce_temp    = args.infonce_temp,
        gumbel_temp     = args.gumbel_temp,
    )

    trainer = InfoGANTrainer(cfg)

    if args.resume:
        ckpt_epoch = trainer.load_checkpoint(args.resume)
        
        start_epoch = args.start_epoch if args.start_epoch is not None else ckpt_epoch + 1
        print(f"[Resume] Will continue from epoch {start_epoch} (total epochs: {cfg.max_epochs})")
        
        trainer.train(start_epoch=start_epoch)
    else:
        trainer.train()


if __name__ == '__main__':
    main()