"""

将俊晖的多个json文件进行merge后，使用语言判别库进行过滤掉非英语的文章，得到最后json文件

"""
import re
import pandas as pd
from langdet.Langdet import Langdet
import glob
import requests
import json
from tqdm import tqdm

dect = Langdet()

def translate(sent):
    try:
        head = {"Content-Type": "application/json; charset=UTF-8"}
        url = 'http://182.92.69.3:7017/niutrans/textTranslation?apikey=3197af0f21041b4eff7e105eb4b1fd94'
        data = {"from": "en", "to": "zh", "src_text": sent, "termDictionaryLibraryId": '', "realmCode":"0",
               "translationMemoryLibraryId": ''}
        data = json.dumps(data)
        data = data.encode("utf-8")
        response = requests.post(url, headers=head, data=data)
        result = json.loads(response.text)
        # print(result)
        resultcn = []
        for datas in result['data']:
            for dt in datas['sentences']:
                resultcn.append(dt['data'])
            zhdata = ''.join(resultcn)
        return zhdata
    except:
        return ''

def merge_json(parent_path):
    # 获取所有CSV文件的路径
    file_paths = glob.glob('{}/*.json'.format(parent_path))

    # 定义一个空的DataFrame来存储合并后的结果
    merged_df = pd.DataFrame()

    # 循环遍历所有CSV文件
    for file_path in file_paths:
        # 读取CSV文件并将其合并到结果DataFrame中
        df = pd.read_json(file_path)
        merged_df = pd.concat([merged_df, df], axis=0, ignore_index=True)

    return merged_df

def filter_en(text):
    if type(text)==list:
        return [True if dect.detect_text(i)==2 else False for i in text]
    else:
        if dect.detect_text(text)==2:
            return True
        else:
            return False


def main():
    df=merge_json("../dataset")
    print(len(df))
    cols=list(df.columns)

    result={k:[] for k in cols}
    result['中文源标题']=[]

    with open('../中文情报.csv', 'w', encoding='utf-8') as f:
        for id,data in tqdm(df.iterrows()):
             if filter_en(data['源标题']):
                 for i in cols:
                     result[i].append(data[i])
                 result['中文原文'][-1]=translate(data['原文'])
                 result['中文源标题'].append(translate(data['源标题']))
                 print(result['源标题'][-1])
                 print(result['中文源标题'][-1])
                 f.write(json.dumps({k: result[k][-1] for k in result.keys()},ensure_ascii=False)+'\n')

    # df_result=pd.DataFrame(data=result)
    # df_result.to_csv('中文情报.csv',index=None)



if __name__=='__main__':
    main()


