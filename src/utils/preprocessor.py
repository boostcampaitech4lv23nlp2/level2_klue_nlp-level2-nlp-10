import pandas as pd
import re


def extract_entity(input_df: pd.DataFrame, tokenizer=None, drop_column=False):
    '''
    데이터셋 DataFrame을 입력받고, entity의 ['word', 'start_idx', 'end_idx', 'type'] 요소들을 추출하여 새 column으로 생성합니다.
    추가되는 새 column들은 다음과 같습니다.
        subject_word, subejct_start_idx, subject_end_idx, subject_type
        object_word, object_start_idx, object_end_idx, object_type
        
    parameters
        tokenizer : transformers.AutoTokenizer를 입력하면 subject_word, object_word가 토큰으로 분리되어 처리됩니다.
        drop_column : 기존 subject_entity, object_entity column들을 drop할지 여부를 결정합니다.
    '''
    df = input_df.copy()
    col_list = ['word', 'start_idx', 'end_idx', 'type']
    sub_contents = [[] for i in range(len(col_list))]
    obj_contents = [[] for i in range(len(col_list))]
    for j, (sub_items, obj_items) in enumerate(zip(df['subject_entity'], df['object_entity'])):
        # 쉼표 중 앞에는 문자가 있고, 뒤에는 공백이 있으며 공백 뒤에 \' 가 있는 쉼표를 기준으로 split합니다.
        sub_items = re.split(r'(?<=\S),\s(?=\')', sub_items[1:-1])
        obj_items = re.split(r'(?<=\S),\s(?=\')', obj_items[1:-1])
        for i, (sub_content, obj_content) in enumerate(zip(sub_items, obj_items)):
            # ':' 중 앞에는 \' 가 있고, 뒤에는 공백이 있는 ':'를 기준으로 split합니다.
            sub_key, sub_insert = map(str.strip, re.split(r'(?<=\'):(?=\s)', sub_content))
            obj_key, obj_insert = map(str.strip, re.split(r'(?<=\'):(?=\s)', obj_content))
            # 문자열의 맨 처음에 위치한 \' 와 맨 뒤에 위치한 \'를 제거합니다.
            sub_insert = re.sub("^\'|\'$", '', sub_insert)
            obj_insert = re.sub("^\'|\'$", '', obj_insert)
            if tokenizer and sub_key == "'word'":
                sub_contents[i].append(tokenizer.tokenize(str(sub_insert)))
                obj_contents[i].append(tokenizer.tokenize(str(obj_insert)))
            else:
                sub_contents[i].append(sub_insert)
                obj_contents[i].append(obj_insert)
                
    # entity의 elements들을 새 column으로 추가합니다.
    prefix_list = ['subject_', 'object_']
    for prefix, contents in zip(prefix_list, [sub_contents, obj_contents]):
        for col, content in zip(col_list, contents):
            col_name = prefix + col
            df[col_name] = content
            if 'idx' in col_name:
                df[col_name] = df[col_name].astype('int')
    if drop_column:
        df = df.drop(['subject_entity', 'object_entity'], axis=1)
    return df