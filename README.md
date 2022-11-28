# Boost2End Relation Extraction 

### 프로젝트 구조
```
├─ config.json
├─ src
│  ├─ main.py
│  ├─ models
│  │  ├─ model.py
│  │  └─ metric.py
│  ├─ train
│  │  └─ train.py
│  ├─ inference
│  │  └─ inference.py
│  └─ utils
│     ├─ __init__.py
│     ├─ base
│     │  └─ dataloader.py
│     ├─ dataloader.py
│     ├─ logger.py
│     └─ parse_config.py
├─ notebooks
|  └─ EDAs.ipynb
├─ data
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