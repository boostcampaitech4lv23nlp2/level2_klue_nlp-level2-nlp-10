
import sys 
import torch 
import argparse 
from utils import ConfigParser


def execute_train(conf, monitor):
    from train import main
    main(conf, monitor)
    return

def execute_inference(conf):   
    from inference import main 
    main(conf)
    return

def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--option', required=True,
                          choices=['train', 'inference'])
    arg_parser.add_argument('--config_path', required=False, default='../config.json')
    arg_parser.add_argument('--monitor', required=False, default=True)
    return arg_parser.parse_args()
    
def main():
    if not torch.cuda.is_available():
        print("Our program supports only CUDA enabled machines")
        sys.exit(1)
    args = get_args()
    conf = ConfigParser().from_json(args.config_path)

    if args.option == 'train':
        execute_train(conf, args.monitor)
    elif args.option == 'inference':
        execute_inference(conf)
    

if __name__ == '__main__':
    main()