import pandas as pd

from load_data import load_content_json
import json
from bart.predict import BartPredictor
from pegusus.predict import PegasusPredictor
from rouge import Rouge

def eval_score(text,title,predictor=None):
    #first sentence

    result=[]

    for i in text:
        result.append(predictor.predict(text, max_length=20))

    assert len(result)==len(title)
    rouge = Rouge()
    rouge_score = rouge.get_scores(result, title)

    print(rouge_score)

def load_file(data_path):
    if data_path.split('.')[-1] == 'csv':
        df = pd.read_csv(data_path)
    elif data_path.split('.')[-1] == 'json':
        df = pd.read_json(data_path)

    return df['中文标题'].to_list(), df['整编内容'].to_list()

def main():
    title_list,text_list=load_file("../data/dataset/validation.json")

    bart_predictor = BartPredictor('../pretrain_model/bart-base')
    pegasus_predictor = PegasusPredictor('../pretrain_model/pegasus_238M')

    print('bart score')
    eval_score(text_list,title_list,bart_predictor)

    print('pegasus score')
    eval_score(text_list,title_list,pegasus_predictor)

if __name__=="__main__":
    main()