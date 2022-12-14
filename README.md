# π Boost2End Relation Extraction Task π

> ## π **Boost2End νμ΄ μ΄λ² νλ‘μ νΈμμ λ¬μ±ν κ²**
> - **μ¬μ¬μ©μ± λ° νμμ μν νλ‘μ νΈ μ‘°μ±**
> - **λ°μ΄ν° μ μ²λ¦¬** λ° **Entity Tagging**, μΆκ°μ μΈ Vocabμ μ μ©ν **Customized BERT Wordpiece Tokenizerλ₯Ό** ν΅ν μ±λ₯ ν₯μ λλͺ¨ 
> - λ°μ΄ν°μμ κ·Ήλ¨μ  λΆκ· νκ³Ό val loss μλ ΄ λ¬Έμ λ₯Ό ν΄κ²°νκΈ° μν **Focal Loss**, **Stratified k-fold** κΈ°λ² μ μ©
> - κ²°κ³Όλ¬Όμ ν λλ‘ **soft voting μμλΈ κΈ°λ²** μ μ© 




## 1. νλ‘μ νΈ κ°μ
- **νλ‘μ νΈ μ£Όμ ** : λ¬Έμ₯ λ΄ κ°μ²΄κ° κ΄κ³ μΆμΆ(Relation Extraction)
    
- **νλ‘μ νΈ κ°μ** : λ¬Έμ₯ μ λ λ¨μ΄κ°μ κ΄κ³λ₯Ό 30κ°μ§ ν΄λμ€ μ€ νλλ‘ λΆλ₯. λ¬Έμ₯μ μν€νΌλμ, μν€νΈλ¦¬, policy_breifingμμ μΆμΆλμμ
    
- **νλ‘μ νΈ κΈ°κ°** : 11/14 ~ 12/1
- **νμ© μ₯λΉ λ° μ¬λ£**
    - GPU : v100 * 5
    - νμ ν΄ : Github, Notion, Wandb
    - κ°λ° νκ²½ : Ubuntu 18.04
- **κΈ°λν¨κ³Ό**
    - λ¨μ΄κ°μ κ΄κ³μ± νμμ λ¬Έμ₯μ μλ―Έλ μλλ₯Ό ν΄μνλ κ²μ λμμ μ£Όμ΄ λ€μν μμ°μ΄ μ²λ¦¬ μλΉμ€λ‘μ νμ₯μ΄ κ°λ₯ν¨ ex) QA μμ€ν
    - κ΄κ³ μΆμΆμ μ§μ κ·Έλν κ΅¬μ±μ μ¬μ©λ  μ μμΌλ©°, λΉκ΅¬μ‘°μ μΈ μμ°μ΄ λ¬Έμ₯μμ κ΅¬μ‘°μ μΈ tripletμ μΆμΆν΄ μ μ©ν μ λ³΄λ‘ νμ©ν  μ μμ

- **νλ‘μ νΈ κ΅¬μ‘°**
        
    ```bash
        ββ config.yaml
        ββ src
        β  ββ main.py
        β  ββ models
        β  β  ββ model.py
        β  β  ββ losses.py
        β  β  ββ metric.py
        β  ββ train
        β  β  ββ train.py
        β  β  ββ train_stratified_kfold.py
        β  β  ββ train_stratified_onefold.py
        β  ββ inference
        β  β  ββ inference.py
        β  ββ utils
        β     ββ __init__.py
        β     ββ base
        β     β  ββ dataloader.py
        β     ββ dataloader.py
        β     ββ logger.py
        β     ββ preprocessor.py
        β     ββ prepare_custom_tokenizer.py
        ββ notebooks
        β  ββ EDAs.ipynb
        ββ data
        ββ tokenizers
        β  ββ klue_bert_base
        β  ββ klue_roberta_large
        ββ unit_test
        β  ββ entity_special_token.ipynb
        β  ββ test_stratified_kfold.ipynb
        β  ββ Testcode_Datacleansing .ipynb
        ββ requirements.txt
        ββ .gitignore
    ```
        
- **λ°μ΄ν°μ κ΅¬μ‘°**
        
    ```jsx
        id : κ° λ°μ΄ν°μ κ³ μ  id
        sentence : κ΄κ³ μΆμΆ λμμΈ λ λ¨μ΄κ° ν¬ν¨λ μ¬λ¬ λ¬Έμ₯λ€
        subject_entity : subject λ¨μ΄. word, start_idx, end_idx, typeμ΄ dict ννλ‘ μ£Όμ΄μ§λ€.
          ββ word : λ¨μ΄
          ββ start_idx : λ¬Έμ₯ λ΄ λ¨μ΄κ° μμνλ index
          ββ end_idx : λ¬Έμ₯ λ΄ λ¨μ΄κ° λλλ index
          ββ type : λ¨μ΄μ νμ. λ¨μ΄μ λ°λΌ PER, ORG, DAT, λ±μ΄ μλ€.
        object_entity : object λ¨μ΄. νμμ subject_entityμ λμΌνλ€.
        label : λ λ¨μ΄κ°μ κ΄κ³. μ΄ 30κ°μ§λ‘ κ΅¬μ±λλ€.
        source : λ°μ΄ν° μΆμ².
    ```
    


## Contributors

- **κΉλ¨κ· [(Github)](https://github.com/manstar1201) : λ°μ΄ν° ν**
    - EDA
    - λ¬Έμ₯ λ΄ μ€λ³΅ λ° μμ€ λ°μ΄ν° μ μ²λ¦¬
    - Masked Language Modelingμ ν΅ν λ°μ΄ν° μ¦κ°
    
- **κΉμ° [(Github)](https://github.com/jtlsan) : λͺ¨λΈλ§ ν**
    - μ£Όμ΄μ§ λ°μ΄ν°μμ λͺ¨λΈμ μ μμν€κΈ° μν domain adaptation κ΅¬ν
    - KE-T5 λͺ¨λΈμ μ¬μ©ν μ±λ₯ λΉκ΅
    - soft-votingμ μ μ©ν μμλΈ κ΅¬ν
    
- **μμ£ΌμΈ [(Github)](https://github.com/EJueon) : νλ‘μ νΈ μ μ²΄ κ΅¬μ±, μ½λ λ° μ΄μ κ΄λ¦¬ν, νμ΅ κ°μ ν**
    - νμ λ° μ¬μ¬μ©μ±μ μν νλ‘μ νΈ μ½λ κ΅¬μ± / νμμ μν ννλ¦Ώ κ΅¬μ±
    - wandb μ€μ  λ° scheduler, checkpoint λ±μ κ΅¬νμ ν΅ν μ€ν νκ²½ μ‘°μ±
    - νλ‘μ νΈ μ½λμ μ λ°μ μΈ μ΄μμ μ½λ κ΄λ¦¬λ₯Ό μ§ννμμ
    - νμ΅ κ°μ μ μν stratified-kfold κΈ°λ²κ³Ό focal loss κΈ°λ² μ μ©
    
- **μ΄λμ°¬ [(Github)](https://github.com/DongChan-Lee) : λ°μ΄ν° ν, μ½λ λ° μ΄μ κ΄λ¦¬ν, μ€ν κ΄λ¦¬ν**
    - μ μ²΄μ μΈ νλ‘μ νΈ μ€ν κ³ν μλ¦½ λ° μ±λ₯ κ°μ μ μν μμ΄λμ΄ μ μ
    - KLUE RE λ°μ΄ν°μμ λ§κ² BERT wordpiece vocabμ customizingνμ¬ [UNK] tokenμ΄ λ°μνμ§ μλλ‘ μ²λ¦¬
    - μ΅μ’μ μΌλ‘ κ°μκ° μ§νν λΆλΆμ ν©μ³μ λͺ¨λΈλ§
    - λ°μ΄ν° EDA λ° μ μ²λ¦¬ λ°©ν₯ μ€μ 
    - Linux Serverμμ Anaconda κ°μνκ²½ μλ μ€μ νλ shell script κ΅¬μ±νμ¬ μ€ν κ΄λ¦¬ μ©μ΄μ± νλ³΄
    
- **μ΄μ ν [(Github)](https://github.com/Jlnus) : λ°μ΄ν° ν**
    - λ°μ΄ν° EDAλ₯Ό ν΅ν΄ entity typeκ³Ό labelμ κ΄κ³ νμΈ
    - νμ΅ μ±λ₯ ν₯μμ μν΄ entity type μ λ³΄λ₯Ό taggingνμ¬ μ μ²λ¦¬

## How to use
```shell
# train version
python main.py --opt=train --version=[λͺ¨λΈ νμΌ μ΄λ¦]

# one-fold train version
python main.py --opt=train_stratified --version=[λͺ¨λΈ νμΌ μ΄λ¦]

# inference version
python main.py --opt=inference --model_path=[λͺ¨λΈ κ²½λ‘] --version=[submission νμΌ μ΄λ¦]
```

### [UNK] tokenμ μ μΈν custom tokenizer μ¬μ©λ²
```yaml
# config.yaml

# custom tokenizer κ²½λ‘ μ€μ  (`tokenizers` directory)
tokenizer_name: "../tokenizers/klue_roberta_large"

# μΌλ° huggingface tokenizer
tokenizer_name: "klue/roberta-large"
```
