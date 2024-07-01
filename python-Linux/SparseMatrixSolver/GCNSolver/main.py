import argparse
import os
import sys
import time

from nsls.__main__ import Solve

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural network solver')

    parser.add_argument('--config', type=str, default='./config/nsls_stand_small_128.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/epoch=49-step=312499.ckpt', help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default='./logs/result.log', help='Path to output file')


    args = parser.parse_args()

    start_time = time.time()
    Solve(args.config, args.checkpoint, args.output)
    end_time = time.time()
    print("Total time: {} ms".format((end_time - start_time)*1000))

