from load_data import load_content_json
from extract_summary.chinese_sentence_cut import cut_sent
from extract_summary.cn_textrank import  textRank
from extract_summary.mmr import mmr

from bart.predict import BartPredictor
from pegusus.predict import PegasusPredictor


def summary(text,title,bart_predictor=None,pegasus_predictor=None):
    #first sentence

    result=[]
    result.append('title :'+title)

    print('first sentence :')
    t=cut_sent(text)[0]
    print('first sentence :',t)
    result.append('first sentence :'+t)

    print('textrank :')
    t=textRank(text,nums=1)
    print(t)
    result.append('text rank :'+ t)

    print('mmr :')
    t=mmr(text,num=1)
    print(t)
    result.append('mmr :'+ t)

    if bart_predictor != None:
        print('bart :')
        t = bart_predictor.predict(text,max_length=20)
        print(t)
        result.append('bart :'+t)

    if pegasus_predictor != None:
        print('pegasus :')
        t = pegasus_predictor.predict(text,max_length=20)
        print(t)
        result.append('pegasus :'+ t)

    return  result


def main():
    content_json=load_content_json("../data/train.csv")

    bart_predictor = BartPredictor('../pretrain_model/bart-base')
    pegasus_predictor = PegasusPredictor('../pretrain_model/pegasus_238M')

    nums=20

    with open('result.txt','w',encoding='utf-8') as f:
        for i in range(nums):
            result=summary(content_json["content_cn"][i],content_json["title"][i],bart_predictor=bart_predictor,pegasus_predictor=pegasus_predictor)
            for j in result:
                f.write(j+'\n')
            f.write('\n')

if __name__=="__main__":
    main()