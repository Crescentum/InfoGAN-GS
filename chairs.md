python train_mnist.py --epochs 50 

python train_mnist.py --epochs 50 --gumbel_temp 0.5

python visualize.py --ckpt checkpoints/chairs_vanilla_final.pt --dataset chairs --out_dir results_chairs

