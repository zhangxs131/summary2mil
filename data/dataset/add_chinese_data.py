import json
from code.translate_utils.TranslateModelByYdWeb import translationModelByYoudao
import sys
from tqdm import tqdm

def trans_chinese(file_name,save_name):

    with open(file_name,'r',encoding='utf-8') as f:
        result=json.loads(f.read())


    result_list=[]
    for id,i in tqdm(enumerate(result)):
        # try:
            translated=translationModelByYoudao(i['原文'])
            result[id]['中文原文']=translated
            print(translated)
            result_list.append(result[id])
        # except:
        #     print('error_{}'.format(id))
        #     continue

    with open(save_name,'w',encoding='utf-8') as f:
        f.write(json.dumps(result_list,ensure_ascii=False))

def main():
    file_name='train.json'
    save_name='train_1.json'
    trans_chinese(file_name,save_name)

if __name__=='__main__':
    main()