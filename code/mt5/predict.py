from transformers import MT5ForConditionalGeneration, MT5Tokenizer

def predict(model,tokenizer,text,max_length=20):
# 使用分词器将文本转换为MT5模型所需的编码形式

    prompt="summarize english to chinese:"
    text=prompt+text
    inputs = tokenizer.encode(text, return_tensors="pt",padding=True)
    # 生成摘要
    summary_ids = model.generate(inputs,
                                  max_length=max_length, # 摘要的最大长度
                                  num_beams=4, # 生成摘要时使用的beam search的数量
                                  length_penalty=2.0, # 控制生成摘要时长度惩罚的系数
                                  no_repeat_ngram_size=3, # 控制生成摘要时避免出现重复n-gram的大小
                                  early_stopping=True # 控制生成摘要时是否进行早停策略
                                 )

    t = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return t
def main():
    pretrain_model_path = "../../pretrain_model/mt5-base"
    model = MT5ForConditionalGeneration.from_pretrained(pretrain_model_path)
    tokenizer = MT5Tokenizer.from_pretrained(pretrain_model_path)

    text = "在北京冬奥会自由式滑雪女子坡面障碍技巧决赛中，中国选手谷爱凌夺得银牌。祝贺谷爱凌！今天上午，自由式滑雪女子坡面障碍技巧决赛举行。决赛分三轮进行，取选手最佳成绩排名决出奖牌。第一跳，中国选手谷爱凌获得69.90分。在12位选手中排名第三。完成动作后，谷爱凌又扮了个鬼脸，甚是可爱。第二轮中，谷爱凌在道具区第三个障碍处失误，落地时摔倒。获得16.98分。网友：摔倒了也没关系，继续加油！在第二跳失误摔倒的情况下，谷爱凌顶住压力，第三跳稳稳发挥，流畅落地！获得86.23分！此轮比赛，共12位选手参赛，谷爱凌第10位出场。网友：看比赛时我比谷爱凌紧张，加油！"
    text_en="In the women's freestyle skiing slopestyle final at the Beijing Winter Olympics, Chinese athlete Gu Ailing won the silver medal. Congratulations to Gu Ailing! The final was held this morning and consisted of three rounds, with the best result of each athlete used to determine the ranking and award medals. In the first jump, Gu Ailing scored 69.90 points, ranking third among the 12 contestants. After completing her performance, she made a cute face, adding to her charm. In the second round, Gu Ailing made a mistake at the third obstacle in the prop area and fell to the ground, earning 16.98 points. Netizens encouraged her, saying it's okay to fall, keep going! Despite falling in the second jump, Gu Ailing withstood the pressure and performed steadily on the third jump, landing smoothly and scoring 86.23 points. In this round of competition, a total of 12 contestants participated, and Gu Ailing was the 10th to appear. Netizens expressed that they were more nervous than Gu Ailing while watching the competition and encouraged her to keep going."
    result=predict(model,tokenizer,text_en)
    print(result)

if __name__=="__main__":
    main()
