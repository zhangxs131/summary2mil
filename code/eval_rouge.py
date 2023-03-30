import pandas as pd
from tqdm import tqdm
from bart.predict import BartPredictor
from pegusus.predict import PegasusPredictor
from rouge import Rouge

def cut(obj, sec):
    return [obj[i:i+sec] for i in range(0,len(obj),sec)]

def eval_score(text,title,predictor=None,eval_batch=32):
    #first sentence

    rouge = Rouge()
    result=[]
    text_list=cut(text,eval_batch)
    title=[(' ').join(list(i)) for i in title]
    title_list=cut(title,eval_batch)
    for i in range(len(text_list)):
        print(i)
        result_t=predictor.predict(text_list[i], max_length=20)
        result_t=[(' ').join(list(i)) for i in result_t]
        rouge_score = rouge.get_scores(result_t, title_list[i],avg=True)
        print(rouge_score)
        result+=result_t

    assert len(result)==len(title)
    rouge_score = rouge.get_scores(result, title,avg=True)

    print('full score:')
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

    print('zero shot bart score')
    eval_score(text_list,title_list,bart_predictor)

    print('zero shot pegasus score')
    eval_score(text_list,title_list,pegasus_predictor)

    fine_tuned_pegasus_predictor=PegasusPredictor('./pegusus/results/checkpoint-1328')

    print('zero shot pegasus score')
    eval_score(text_list, title_list, fine_tuned_pegasus_predictor)

if __name__=="__main__":
    main()