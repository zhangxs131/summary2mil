from rouge import Rouge

text=['我 是 张 晓 松','我 是 小 羊']
title=['张 晓 松','小 羊']

t1='张 晓 松'
t2='小 羊 松'
rouge = Rouge()
rouge_score = rouge.get_scores(text, title,avg=True)
print(rouge_score)