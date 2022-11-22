
import sys 
import torch 
import argparse 
from omegaconf import OmegaConf
from utils import ConfigParser
from utils import set_seed, print_msg


def execute_train(conf, version, is_monitor, is_scheduler):
    from train.train import main
    main(conf, version, is_monitor, is_scheduler)
    return

def execute_inference(conf, version, model_path, is_checkpoint = False):   
    from inference.inference import main 
    main(conf, version, model_path, is_checkpoint)
    return
    

def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--option', required=True,
                          choices=['train', 'inference', 'voting'], help='학습, 추론 모드 설정')
    arg_parser.add_argument('--config_path', required=False, default='../config.json', help='configuration 파일 경로')
    arg_parser.add_argument('--version', required=False, default='v0', help='저장시 사용하는 부가 파일명')
    arg_parser.add_argument('--model_path', required=False, default=None, help='추론시 사용하는 모델 파일 경로')
    arg_parser.add_argument('--is_monitor', required=False, default=True, help='wandb 사용 여부 판단')
    arg_parser.add_argument('--is_scheduler', required=False, default=True, help='scheduler 사용 여부 판단')
    arg_parser.add_argument('--is_checkpoint', required=False, default=False, help='checkpoint 파일 사용 여부 판단')
    arg_parser.add_argument('--submission_files', default=[], nargs='+', help='Ensemble voting submissions')
    return arg_parser.parse_args()
    
def main():
    if not torch.cuda.is_available():
        print("Our program supports only CUDA enabled machines")
        sys.exit(1)
    args = get_args()
    conf = OmegaConf.load(args.config_path)
    
    set_seed(conf.seed) # random seed 설정
    if args.option == 'train':
        execute_train(conf, args.version, args.is_monitor, args.is_scheduler)
    elif args.option == 'inference':
        if not args.model_path:
            print_msg("model 경로를 찾을 수 없습니다. argument --model_path에 모델 경로를 작성해주세요. ", "ERROR")
            return
        execute_inference(conf, args.version, args.model_path, args.is_checkpoint)
    elif args.option == 'voting':
        execute_voting(conf, args.submission_files)

if __name__ == '__main__':
    main()