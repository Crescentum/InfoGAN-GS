import argparse
import torch

from trainer_celeba import InfoGANTrainer, TrainerConfig, VALID_MODES


def parse_args():
    p = argparse.ArgumentParser(description="CelebA InfoGAN training")
    p.add_argument("--mode", type=str, default="vanilla",
                   choices=list(VALID_MODES))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--updates_per_epoch", type=int, default=0,
                   help="0 means one full pass over the DataLoader")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--log_dir", type=str, default="./logs/celeba_single_code")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints/celeba_single_code")
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lambda_disc", type=float, default=1.0)
    p.add_argument("--lambda_cont", type=float, default=0.0)
    p.add_argument("--n_critic", type=int, default=1,
                   help="discriminator updates per generator update")
    p.add_argument("--resume", type=str, default=None,
                   help="path to .pt checkpoint to resume from")
    p.add_argument("--start_epoch", type=int, default=None,
                   help="override start epoch when resuming")
    return p.parse_args()


def main():
    args = parse_args()

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        resumed_mode = ckpt.get("mode", args.mode)
        resumed_epoch = ckpt.get("epoch", -1)
        if resumed_mode not in VALID_MODES:
            raise ValueError(
                f"CelebA currently supports only {VALID_MODES}; "
                f"checkpoint mode is '{resumed_mode}'."
            )
        args.mode = resumed_mode
        print(f"[Resume] Checkpoint: dataset=celeba, mode={resumed_mode}, epoch={resumed_epoch}")

    cfg = TrainerConfig(
        mode=args.mode,
        dataset="celeba",
        max_epochs=args.epochs,
        updates_per_epoch=args.updates_per_epoch,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        checkpoint_dir=args.ckpt_dir,
        lr_d=args.lr_d,
        lr_g=args.lr_g,
        lambda_disc=args.lambda_disc,
        lambda_cont=args.lambda_cont,
        n_critic=args.n_critic,
    )

    trainer = InfoGANTrainer(cfg)
    if args.resume:
        ckpt_epoch = trainer.load_checkpoint(args.resume)
        start_epoch = args.start_epoch if args.start_epoch is not None else ckpt_epoch + 1
        print(f"[Resume] Will continue from epoch {start_epoch} (total epochs: {cfg.max_epochs})")
        trainer.train(start_epoch=start_epoch)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
