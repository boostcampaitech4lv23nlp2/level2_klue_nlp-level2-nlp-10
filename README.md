# Boost2End Relation Extraction 

### 프로젝트 구조
```
├─ config.yaml
├─ src
│  ├─ main.py
│  ├─ models
│  │  ├─ model.py
│  │  └─ metric.py
│  ├─ train
│  │  └─ train.py
│  │  └─ train_stratified_kfold.py
│  │  └─ train_stratified_onefold.py
│  ├─ inference
│  │  └─ inference.py
│  └─ utils
│     ├─ __init__.py
│     ├─ base
│     │  └─ dataloader.py
│     ├─ dataloader.py
│     ├─ logger.py
│     └─ preprocessor.py
│     └─ prepare_custom_tokenizer.py
├─ notebooks
|  └─ EDAs.ipynb
├─ data
├─ tokenizers
│  ├─ klue_bert_base
│  │  └─ config.json
│  │  └─ special_tokens_map.json
│  │  └─ tokenizer_config.json
│  │  └─ tokenizer.json
│  │  └─ vocab.txt
│  └─ klue_roberta_large
│     └─ config.json
│     └─ special_tokens_map.json
│     └─ tokenizer_config.json
│     └─ tokenizer.json
│     └─ vocab.txt
├─ unit_test
│  ├─ entity_special_token.ipynb
│  ├─ test_stratified_kfold.ipynb
│  └─ Testcode_Datacleansing .ipynb
└─ requirements.txt
└─ .gitignore
```

### 실행 방법
```shell
# train version
python main.py --opt=train --version=[모델 파일 이름]

# one-fold train version
python main.py --opt=train_stratified --version=[모델 파일 이름]

# inference version
python main.py --opt=inference --model_path=[모델 경로] --version=[submission 파일 이름]
```

### [UNK] token을 제외한 custom tokenizer 사용법
```yaml
# config.yaml

# custom tokenizer 경로 설정 (`tokenizers` directory)
tokenizer_name: "../tokenizers/klue_roberta_large"

# 일반 huggingface tokenizer
tokenizer_name: "klue/roberta-large"
```