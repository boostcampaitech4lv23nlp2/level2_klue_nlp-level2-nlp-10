import re
import pandas as pd
import numpy as np

TYPE_TOKENS = dict(
    subject_per_start="<subj:PER>",
    subject_org_start="<subj:ORG>",
    subject_per_end="</subj:PER>",
    subject_org_end="</subj:ORG>",
    object_per_start="<obj:PER>",
    object_org_start="<obj:ORG>",
    object_loc_start="<obj:LOC>",
    object_dat_start="<obj:DAT>",
    object_poh_start="<obj:POH>",
    object_noh_start="<obj:NOH>",
    object_per_end="</obj:PER>",
    object_org_end="</obj:ORG>",
    object_loc_end="</obj:LOC>",
    object_dat_end="</obj:DAT>",
    object_poh_end="</obj:POH>",
    object_noh_end="</obj:NOH>",
)

def fix_entity_index(
    sentence: str, subject_entity: str, object_entity: str
    ) -> tuple:
    """
    전처리가 완료된 문장 sentence와 원본 데이터의 subject_entity, object_entity를 입력으로 받습니다.
    전처리로 인해 위치가 바뀐 subject_word와 object_word의 start_index와 end_index를 수정하여 수정된 (subject_entity, object_entity)의 tuple을 리턴합니다.
    """
    # str의 entity를 dict로 변경합니다.
    dict_subject = eval(subject_entity)
    dict_object = eval(object_entity)
    
    # entity dict에 속한 값을 가져옵니다.
    s_word, s_start, s_end, s_typ = dict_subject.values()
    o_word, o_start, o_end, o_typ = dict_object.values()
    
    # 전처리된 문장 sentence에서 subject entity의 word를 찾아 matched에 저장합니다.
    # finditer를 사용하는 이유는 찾고자 하는 word가 sentence내에 여러개 있을 수 있기 때문입니다.
    matched = re.finditer(re.escape(s_word), sentence)
    new_standard = float("inf")

    # 매치된 문장 내 단어 정보를 순회합니다.
    for word in matched:
        # 전처리 이전 문장의 start_index(변수명: s_start)를 기준으로, 전처리된 문장 내 word 위치와의 차이를 계산하여 차이가 가장 작은 단어를 탐색합니다
        start_standard = abs(s_start - word.span()[0])
        if new_standard >= start_standard:
            s, e = word.span()[0], word.span()[1] - 1
            new_standard = start_standard
    dict_subject['start_idx'], dict_subject['end_idx']= s, e

    # 동일한 작업을 dict_object에도 수행합니다.
    matched = re.finditer(re.escape(o_word), sentence)
    new_standard = float("inf")

    for word in matched:
        start_standard = abs(o_start - word.span()[0])
        if new_standard >= start_standard:
            s, e = word.span()[0], word.span()[1] - 1
            new_standard = start_standard
    dict_object['start_idx'], dict_object['end_idx']= s, e
    
    dict_subject = str(dict_subject)
    dict_object = str(dict_object)
    
    return (dict_subject, dict_object)

def preprocess_double_quotation(sts: str) -> str:
    """
    큰따옴표가 중복으로 사용될 경우 문장을 전처리하여 리턴합니다.
    큰따옴표 사이에 정보가 있을 경우에는 단순 중복 큰따옴표만 단일 큰따옴표로 교체합니다.
    괄호로 둘러쌓여있거나 큰따옴표 사이에 정보가 없을 경우 해당 부분을 삭제합니다.
    """
    result = sts

    matched_list = re.findall('"".*?""', result)
    for word in matched_list:
        result = result.replace(word, word[1:-1])

    matched_list = re.findall('\(""\)', result)
    for word in matched_list:
        result = result.replace(word, '')

    matched_list = re.findall('""', result)
    for word in matched_list:
        result = result.replace(word, '')

    matched_list = re.findall('  ', result)
    for word in matched_list:
        result = result.replace(word, ' ')

    return result

def preprocess_overlapped_bracket(sts: str) -> str:
    """
    중복된 내용을 담고 있는 괄호가 사용될 경우 중복된 내용을 삭제하여 문장을 전처리하여 리턴합니다.
    """
    result = sts

    matched_list = re.finditer('\([^\(\)]+\)', result)
    idx = ()
    word = ''

    for match in matched_list:
        idx_2 = match.span()
        word_2 = match.group()
        if word == word_2 and idx[1] == idx_2[0]:
            result = result.replace(word * 2, word)
        idx = idx_2
        word = word_2

    return result

def preprocess_double_sqbracket(sts: str) -> str:
    """
    사각형 괄호가 중복으로 사용될 경우 단일 사각형 괄호로 교체하여 문장을 전처리하여 리턴합니다.
    """
    result = sts
    result = result.replace('[[', '[')
    result = result.replace(']]', ']')

    return result

def preprocess_spaced_comma(sts: str) -> str:
    """
    양쪽 공백으로 둘러 쌓인 쉼표와 중복 쉼표가 사용될 경우 앞 공백을 삭제하거나 단일 쉼표로 교체하여 문장을 전처리하여 리턴합니다.
    """
    result = sts
    result = result.replace(" , ", ", ")
    result = result.replace(",,", ",")

    return result

def preprocess_double_space(sts: str) -> str:
    """
    중복 공백이 사용될 경우 단일 공백으로 교체하여 문장을 전처리하여 리턴합니다.
    """
    result = sts
    result = result.replace("  ", " ")

    return result

def preprocess_double_dash(sts: str) -> str:
    """
    중복 대쉬 기호가 사용될 경우 단일 대쉬 기호로 교체하여 문장을 전처리하여 리턴합니다.
    """
    result = sts
    result = result.replace("--", "-")

    return result

def data_cleansing(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    전체 DataFrame을 입력으로 받고 각 row마다 문장을 전처리하고 entity 정보를 수정하여 수정된 DataFrame을 리턴합니다.
    """
    df = input_df.copy()

    for row in df.iterrows():
        idx, sentence, subject_entity, object_entity = row[1].id, row[1].sentence, row[1].subject_entity, row[1].object_entity

        sentence = preprocess_double_quotation(sentence)
        sentence = preprocess_double_sqbracket(sentence)
        sentence = preprocess_spaced_comma(sentence)
        sentence = preprocess_double_dash(sentence)
        sentence = preprocess_overlapped_bracket(sentence)
        sentence = preprocess_double_space(sentence)
        subject_entity, object_entity = fix_entity_index(sentence, subject_entity, object_entity)

        df.loc[idx, 'sentence'] = sentence
        df.loc[idx, 'subject_entity'] = subject_entity
        df.loc[idx, 'object_entity'] = object_entity

    return df
    
def extract_entity(
    input_df: pd.DataFrame, tokenizer=None, drop_column=False
) -> pd.DataFrame:
    """
    데이터셋 DataFrame을 입력받고, entity의 ['word', 'start_idx', 'end_idx', 'type'] 요소들을 추출하여 새 column으로 생성합니다.
    추가되는 새 column들은 다음과 같습니다.
        subject_word, subejct_start_idx, subject_end_idx, subject_type
        object_word, object_start_idx, object_end_idx, object_type

    parameters
        tokenizer : transformers.AutoTokenizer를 입력하면 subject_word, object_word가 토큰으로 분리되어 처리됩니다.
        drop_column : 기존 subject_entity, object_entity column들을 drop할지 여부를 결정합니다.
    """
    df = input_df.copy()
    col_list = ["word", "start_idx", "end_idx", "type"]
    sub_contents = [[] for _ in range(len(col_list))]
    obj_contents = [[] for _ in range(len(col_list))]

    for sub_items, obj_items in zip(df["subject_entity"], df["object_entity"]):
        # 쉼표 중 앞에는 문자가 있고, 뒤에는 공백이 있으며 공백 뒤에 \' 가 있는 쉼표를 기준으로 split합니다.
        sub_items = re.split(r"(?<=\S),\s(?=\')", sub_items[1:-1])
        obj_items = re.split(r"(?<=\S),\s(?=\')", obj_items[1:-1])
        for i, (sub_content, obj_content) in enumerate(zip(sub_items, obj_items)):
            # ':' 중 앞에는 \' 가 있고, 뒤에는 공백이 있는 ':'를 기준으로 split합니다.
            sub_key, sub_insert = map(
                str.strip, re.split(r"(?<=\'):(?=\s)", sub_content)
            )
            obj_key, obj_insert = map(
                str.strip, re.split(r"(?<=\'):(?=\s)", obj_content)
            )
            # 문자열의 맨 처음에 위치한 \' 와 맨 뒤에 위치한 \'를 제거합니다.
            sub_insert = re.sub("^'|'$", "", sub_insert)
            obj_insert = re.sub("^'|'$", "", obj_insert)
            if tokenizer and sub_key == "'word'":
                sub_contents[i].append(tokenizer.tokenize(str(sub_insert)))
                obj_contents[i].append(tokenizer.tokenize(str(obj_insert)))
            else:
                sub_contents[i].append(sub_insert)
                obj_contents[i].append(obj_insert)

    # entity의 elements들을 새 column으로 추가합니다.
    prefix_list = ["subject_", "object_"]
    for prefix, contents in zip(prefix_list, [sub_contents, obj_contents]):
        for col, content in zip(col_list, contents):
            col_name = prefix + col
            df[col_name] = content
            if "idx" in col_name:
                df[col_name] = df[col_name].astype("int")
    if drop_column:
        df = df.drop(["subject_entity", "object_entity"], axis=1)
    return df

def entity_tagging(
    dataset: pd.DataFrame, 
) -> pd.DataFrame:
    """
    sentence에 entity type token을 추가하여 전처리합니다.
    -  〈Something〉는 <obj:PER>조지 해리슨</obj:PER>이 쓰고 <subj:ORG>비틀즈</subj:ORG>가 1969년 앨범
    """
    df = dataset.copy()
    data_np = dataset.to_numpy() # numpy로 데이터 탐색
    data_np = np.transpose(data_np)
    
    # data_np의 인덱스입니다
    # 0:'id', 1:'sentence', 2:'label', 3:'source', 4:'subject_word', 5:'subject_start_idx',
    # 6:'subject_end_idx', 7:'subject_type', 8:'object_word', 9:'object_start_idx',
    # 10:'object_end_idx', 11:'object_type')

    for i in df.index:
        subject_start_marker = f"<subj:{data_np[7][i]}>"
        subject_end_marker   = f"</subj:{data_np[7][i]}>"
        object_start_marker  = f"<obj:{data_np[11][i]}>"
        object_end_marker    = f"</obj:{data_np[11][i]}>"
        
        if "'" in data_np[8][i]: # (')이 포함된 문장 전처리
            data_np[8][i] = data_np[8][i].strip('"')
            
        sent = data_np[1][i]
        
        if data_np[5][i] < data_np[9][i]:  # subject가 object보다 앞에 있는 겨우
            tmp = (sent[0:data_np[5][i]] + subject_start_marker + data_np[4][i] + subject_end_marker +
                   sent[data_np[6][i]+1:data_np[9][i]] + object_start_marker + data_np[8][i] + object_end_marker + sent[data_np[10][i]+1:-1])
            
            data_np[5][i] += len(subject_start_marker) # subject_start_idx
            data_np[6][i] += len(subject_start_marker) # subject_end_idx
            data_np[9][i] += len(subject_start_marker + subject_end_marker + object_start_marker) # object_start_idx
            data_np[10][i] += len(subject_start_marker + subject_end_marker + object_start_marker) # object_end_idx
            
        else:  # object가 subject보다 앞에 있는 경우
            tmp = (sent[0:data_np[9][i]] + object_start_marker + data_np[8][i] + object_end_marker +
                   sent[data_np[10][i]+1:data_np[5][i]] + subject_start_marker + data_np[4][i] + subject_end_marker + sent[data_np[6][i]+1:-1])

            data_np[5][i] += len(object_start_marker + object_end_marker + subject_start_marker) # subject_start_idx
            data_np[6][i] += len(object_start_marker + object_end_marker + subject_start_marker) # subject_end_idx
            data_np[9][i] += len(object_start_marker) # object_start_idx
            data_np[10][i] += len(object_start_marker) # object_end_idx 
            
        str_tmp = "".join(tmp)
        data_np[1][i] = str_tmp # sentence
        
    df["sentence"] = data_np[1]
    df["subject_start_idx"] = data_np[5]
    df["subject_end_idx"] = data_np[6]
    df["object_start_idx"] = data_np[9]
    df["object_end_idx"] = data_np[10]

    return df
