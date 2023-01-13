# 🚀 Boost2End Relation Extraction Task 🚀

> ## 📌 **Boost2End 팀이 이번 프로젝트에서 달성한 것**
> - **재사용성 및 협업을 위한 프로젝트 조성**
> - **데이터 전처리** 및 **Entity Tagging**, 추가적인 Vocab을 적용한 **Customized BERT Wordpiece Tokenizer를** 통한 성능 향상 도모 
> - 데이터셋의 극단적 불균형과 val loss 수렴 문제를 해결하기 위한 **Focal Loss**, **Stratified k-fold** 기법 적용
> - 결과물을 토대로 **soft voting 앙상블 기법** 적용 




## 1. 프로젝트 개요
- **프로젝트 주제** : 문장 내 개체간 관계 추출(Relation Extraction)
    
- **프로젝트 개요** : 문장 속 두 단어간의 관계를 30가지 클래스 중 하나로 분류. 문장은 위키피디아, 위키트리, policy_breifing에서 추출되었음
    
- **프로젝트 기간** : 11/14 ~ 12/1
- **활용 장비 및 재료**
    - GPU : v100 * 5
    - 협업 툴 : Github, Notion, Wandb
    - 개발 환경 : Ubuntu 18.04
- **기대효과**
    - 단어간의 관계성 파악은 문장의 의미나 의도를 해석하는 것에 도움을 주어 다양한 자연어 처리 서비스로의 확장이 가능함 ex) QA 시스템
    - 관계 추출은 지식 그래프 구성에 사용될 수 있으며, 비구조적인 자연어 문장에서 구조적인 triplet을 추출해 유용한 정보로 활용할 수 있음

- **프로젝트 구조**
        
    ```bash
        ├─ config.yaml
        ├─ src
        │  ├─ main.py
        │  ├─ models
        │  │  ├─ model.py
        │  │  ├─ losses.py
        │  │  └─ metric.py
        │  ├─ train
        │  │  ├─ train.py
        │  │  ├─ train_stratified_kfold.py
        │  │  └─ train_stratified_onefold.py
        │  ├─ inference
        │  │  └─ inference.py
        │  └─ utils
        │     ├─ __init__.py
        │     ├─ base
        │     │  └─ dataloader.py
        │     ├─ dataloader.py
        │     ├─ logger.py
        │     ├─ preprocessor.py
        │     └─ prepare_custom_tokenizer.py
        ├─ notebooks
        │  └─ EDAs.ipynb
        ├─ data
        ├─ tokenizers
        │  ├─ klue_bert_base
        │  └─ klue_roberta_large
        ├─ unit_test
        │  ├─ entity_special_token.ipynb
        │  ├─ test_stratified_kfold.ipynb
        │  └─ Testcode_Datacleansing .ipynb
        └─ requirements.txt
        └─ .gitignore
    ```
        
- **데이터셋 구조**
        
    ```jsx
        id : 각 데이터의 고유 id
        sentence : 관계 추출 대상인 두 단어가 포함된 여러 문장들
        subject_entity : subject 단어. word, start_idx, end_idx, type이 dict 형태로 주어진다.
          ├─ word : 단어
          ├─ start_idx : 문장 내 단어가 시작하는 index
          ├─ end_idx : 문장 내 단어가 끝나는 index
          ├─ type : 단어의 형식. 단어에 따라 PER, ORG, DAT, 등이 있다.
        object_entity : object 단어. 형식은 subject_entity와 동일하다.
        label : 두 단어간의 관계. 총 30가지로 구성된다.
        source : 데이터 출처.
    ```
    


## Contributors

- **김남규 [(Github)](https://github.com/manstar1201) : 데이터 팀**
    - EDA
    - 문장 내 중복 및 손실 데이터 전처리
    - Masked Language Modeling을 통한 데이터 증강
    
- **김산 [(Github)](https://github.com/mountinyy) : 모델링 팀**
    - 주어진 데이터셋에 모델을 적응시키기 위한 domain adaptation 구현
    - KE-T5 모델을 사용한 성능 비교
    - soft-voting을 적용한 앙상블 구현
    
- **엄주언 [(Github)](https://github.com/EJueon) : 프로젝트 전체 구성, 코드 및 이슈 관리팀, 학습 개선팀**
    - 협업 및 재사용성을 위한 프로젝트 코드 구성 / 협업을 위한 템플릿 구성
    - wandb 설정 및 scheduler, checkpoint 등의 구현을 통한 실험 환경 조성
    - 프로젝트 코드의 전반적인 이슈와 코드 관리를 진행하였음
    - 학습 개선을 위한 stratified-kfold 기법과 focal loss 기법 적용
    
- **이동찬 [(Github)](https://github.com/DongChan-Lee) : 데이터 팀, 코드 및 이슈 관리팀, 실험 관리팀**
    - 전체적인 프로젝트 실험 계획 수립 및 성능 개선을 위한 아이디어 제안
    - KLUE RE 데이터셋에 맞게 BERT wordpiece vocab을 customizing하여 [UNK] token이 발생하지 않도록 처리
    - 최종적으로 각자가 진행한 부분을 합쳐서 모델링
    - 데이터 EDA 및 전처리 방향 설정
    - Linux Server에서 Anaconda 가상환경 자동 설정하는 shell script 구성하여 실험 관리 용이성 확보
    
- **이정현 [(Github)](https://github.com/Jlnus) : 데이터 팀**
    - 데이터 EDA를 통해 entity type과 label의 관계 확인
    - 학습 성능 향상을 위해 entity type 정보를 tagging하여 전처리

## How to use
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
