import pandas as pd
import pytorch_lightning as pl
import torch.nn.functional as F
from models.metrics import compute_metrics
from models.model import KLUEModel
from utils import *
from utils.dataloader import Dataloader, KLUEDataset


def main(conf, version, model_path, is_checkpoint=False):
    print_config(conf)  # configuration parameter 확인
    save_path = setdir(conf.data_dir, conf.submission_dir, reset=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset & dataloader
    test_dataloader = Dataloader(
        conf.tokenizer_name,
        conf.test_data_path,
        conf.label_to_num_dict_path,
        conf.batch_size,
        is_test=True,
    )
    # load model
    model = KLUEModel(conf, device)
    if is_checkpoint:
        model = model.load_from_checkpoint(model_path)
    else:
        model.load_state_dict(torch.load(model_path))
    model.eval()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=conf.max_epoch,
        log_every_n_steps=1,
        precision=16,
        replace_sampler_ddp=False,
    )

    print_msg("예측을 시작합니다....", "INFO")
    probabilities = trainer.predict(model=model, datamodule=test_dataloader)
    probabilities = torch.cat(probabilities)
    predictions = torch.argmax(probabilities, -1).tolist()
    pred_answers = num_to_label(
        predictions, conf.num_to_label_dict_path
    )  # 숫자로 된 class를 원래 문자열 라벨로 변환.
    print_msg("예측이 완료되었습니다", "INFO")

    # submission 파일 저장
    print_msg("submission 파일을 저장합니다....", "INFO")
    output = pd.read_csv(conf.submission_path)
    output = pd.DataFrame(
        {
            "id": output["id"],
            "pred_label": pred_answers,
            "probs": probabilities.tolist(),
        }
    )
    file_name = make_file_name(
        conf.model_name.replace("/", "_"), version=version, format="csv"
    )
    save_path = os.path.join(save_path, file_name)
    output.to_csv(save_path, index=False)
    print_msg("submission 파일 저장을 완료하였습니다....", "INFO")

