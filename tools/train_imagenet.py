import argparse
import os
import torch
from pth_easy.pytorch_gluoncv_model_zoo import get_model
from pth_easy.solver import build_optimizer

def parse_args():
    parser = argparse.ArgumentParser(description='train pytorch model in imagenet.')
    parser.add_argument('--network', type=str, default='mobilenetv3_small', help="network name")
    parser.add_argument('--output-dir', type=str, default='output', help='output path')
    parser.add_argument('--img-path', type=str, default='/home/xie/data/imagenet_val', help='Path of the images')
    parser.add_argument('--batch-size', type=int, default=32, help='batchsize')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--numworkers', type=int, default=4, help='numworkers')
    args = parser.parse_args()
    return args

def train(cfg):
    device = torch.device(cfg.device)
    model = get_model(cfg.network)
    model.to(device)

    optimizer = build_optimizer(model, cfg.learning_rate, cfg.weight_decay)

def main():
	args = parse_args()
	num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
	args.distributed = num_gpus > 1

	if args.distributed:
		torch.cuda.set_device(args.local_rank)
		torch.distributed.init_process_group(backend="nccl", init_method="env://")

	output_dir = args.output_dir
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	model = train(args)


if __name__ == '__main__':
	main()