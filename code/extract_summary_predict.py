from load_data import load_content_json
from extract_summary.chinese_sentence_cut import cut_sent
from extract_summary.cn_textrank import  textRank
from extract_summary.mmr import mmr

def summary(text):
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


def main():
    content_json=load_content_json("../data/train.csv")
    summary(content_json["content_cn"][0])
    print("the title is ")
    print(content_json["title"][0])

if __name__=="__main__":
    main()