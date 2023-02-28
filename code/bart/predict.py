from transformers import BartTokenizer, BartForConditionalGeneration


class BartPredictor(object):
    def __init__(self,pretrain_model='bart-base'):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(pretrain_model)
        self.tokenizer = BartTokenizer.from_pretrained(pretrain_model)

    def predict(self,text):
        inputs = self.tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)

        # Generate Summary
        summary_ids = self.model.generate(inputs["input_ids"], max_length=80)
        t = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return t


def main():


    bart_predictor=BartPredictor('bart-base')

    text = "在北京冬奥会自由式滑雪女子坡面障碍技巧决赛中，中国选手谷爱凌夺得银牌。祝贺谷爱凌！今天上午，自由式滑雪女子坡面障碍技巧决赛举行。决赛分三轮进行，取选手最佳成绩排名决出奖牌。第一跳，中国选手谷爱凌获得69.90分。在12位选手中排名第三。完成动作后，谷爱凌又扮了个鬼脸，甚是可爱。第二轮中，谷爱凌在道具区第三个障碍处失误，落地时摔倒。获得16.98分。网友：摔倒了也没关系，继续加油！在第二跳失误摔倒的情况下，谷爱凌顶住压力，第三跳稳稳发挥，流畅落地！获得86.23分！此轮比赛，共12位选手参赛，谷爱凌第10位出场。网友：看比赛时我比谷爱凌紧张，加油！"

    result=bart_predictor.predict(text)
    print(result)

if __name__=="__main__":
    main()
# model Output: 滑雪女子坡面障碍技巧决赛谷爱凌获银牌
