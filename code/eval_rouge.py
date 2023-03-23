from load_data import load_content_json

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





def main():
    content_json=load_content_json("../data/train.csv")

    bart_predictor = BartPredictor('../pretrain_model/bart-base')
    pegasus_predictor = PegasusPredictor('../pretrain_model/pegasus_238M')

    nums=20
    print('bart score')
    eval_score(text_list,title_list,bart_predictor)

    print('pegasus score')
    eval_score(text_list,title_list,pegasus_predictor)

if __name__=="__main__":
    main()