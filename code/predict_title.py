from load_data import load_content_json
from extract_summary.chinese_sentence_cut import cut_sent
from extract_summary.cn_textrank import  textRank
from extract_summary.mmr import mmr

from bart.predict import BartPredictor
from pegusus.predict import PegasusPredictor


def summary(text,bart_predictor=None,pegasus_predictor=None):
    #first sentence

    print('first sentence :')
    t=cut_sent(text)[0]
    print(t)

    print('textrank :')
    t=textRank(text,nums=1)
    print(t)

    print('mmr :')
    t=mmr(text,num=1)
    print(t)

    if bart_predictor != None:
        print('bart :')
        t = bart_predictor.predict(text)
        print(t)

    if pegasus_predictor != None:
        print('pegasus :')
        t = pegasus_predictor.predict(text)
        print(t)



def main():
    content_json=load_content_json("../data/train.csv")

    bart_predictor=BartPredictor('bart/bart-base')
    pegasus_predictor=PegasusPredictor('pegusus/pegasus-238M')

    for i in range(5):
        summary(content_json["content_cn"][i],bart_predictor=bart_predictor,pegasus_predictor=pegasus_predictor)
        print("the title is ")
        print(content_json["title"][i])

if __name__=="__main__":
    main()