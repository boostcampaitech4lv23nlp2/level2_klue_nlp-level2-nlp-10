from utils import *
from utils.dataloader import KLUEDataset, Dataloader
from models.model import KLUEModel
from models.metrics import compute_metrics
import pytorch_lightning as pl

# FIXME : 전체적으로 testdata를 실행하지 않는 오류 존재. 수정 필요
def main(conf, version, model_path, is_checkpoint=False):
    print_config(conf) # configuration parameter 확인 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    saved_path = setdir(conf.data_dir, conf.submission_dir, reset=False)
    
    # load dataset & dataloader    
    test_dataloader = Dataloader(conf.model_name, 
                                  conf.test_data_path,
                                  conf.label_to_num_dict_path, 
                                  conf.batch_size,
                                  is_test = True)
    # load model 
    model = KLUEModel(conf, device)
    
    if is_checkpoint:
        model = model.load_from_checkpoint(model_path)
    else:
        model.load_state_dict(torch.load(model_path))
    
    trainer = pl.Trainer(accelerator='gpu', 
                         devices=1, 
                         max_epochs=conf.max_epoch, 
                         log_every_n_steps=1)
    
    print_msg(f'Make predictions....', 'INFO')
    predictions = trainer.predict(model=model, datamodule=test_dataloader)
    print(predictions)
    input('>>>>')
    pred_answer = num_to_label(pred_answers, conf.num_to_label_dict_path) # 숫자로 된 class를 원래 문자열 라벨로 변환.
    print_msg('예측이 완료되었습니다', 'INFO')
    
    print(pred_answer)
    input('>>')
    output = pd.read_csv(conf.submission_path)
    
    
    output['pred_label'] = predictions
    new_output = pd.DataFrame({'id':output['id'],'pred_label':pred_answer,'probs':output_prob,})
    file_name = make_file_name(conf.model_name.replace('/', '_'), version=version, format='csv')
    save_path = os.path.join(save_path, file_name)
    output.to_csv(save_path, index=False)
    