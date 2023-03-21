from transformers import T5Tokenizer,T5ForConditionalGeneration

def predict(model,tokenizer,text):
    inputs = tokenizer(text, return_tensors="pt",truncation=True)

    # Generate Summary
    summary_ids = model.generate(inputs["input_ids"],max_length=80)
    t = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return t

def main():
    pretrain_model_path = "../../pretrain_model/t5-base"
    model = T5ForConditionalGeneration.from_pretrained(pretrain_model_path)
    tokenizer = T5Tokenizer.from_pretrained(pretrain_model_path)

    text = "summarize:在北京冬奥会自由式滑雪女子坡面障碍技巧决赛中，中国选手谷爱凌夺得银牌。祝贺谷爱凌！今天上午，自由式滑雪女子坡面障碍技巧决赛举行。决赛分三轮进行，取选手最佳成绩排名决出奖牌。第一跳，中国选手谷爱凌获得69.90分。在12位选手中排名第三。完成动作后，谷爱凌又扮了个鬼脸，甚是可爱。第二轮中，谷爱凌在道具区第三个障碍处失误，落地时摔倒。获得16.98分。网友：摔倒了也没关系，继续加油！在第二跳失误摔倒的情况下，谷爱凌顶住压力，第三跳稳稳发挥，流畅落地！获得86.23分！此轮比赛，共12位选手参赛，谷爱凌第10位出场。网友：看比赛时我比谷爱凌紧张，加油！"

    result=predict(model,tokenizer,text)
    print(result)

if __name__=="__main__":
    main()
# model Output: 滑雪女子坡面障碍技巧决赛谷爱凌获银牌
