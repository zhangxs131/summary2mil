# -*- coding: utf-8 -*-
# 接口地址https://api.aa1.cn/doc/api-fanyi-yd.html
import requests,json
from nltk import sent_tokenize    #以符号形式实现分句


# 需要前面加一个.才能识别本地模块
#from TranslateModelByNLPIR import translateModelByNLPIR

def translationModelByYD(text):
    if text=='':
        print('text is null')
        return text
    # url = 'https://v.api.aa1.cn/api/api-fanyi-yd/index.php?msg='+text+'&type=2'
    # response = requests.get(url)
    url = 'https://v.api.aa1.cn/api/api-fanyi-yd/index.php'
    par = {
        "msg": text,
        "type": 2
    }
    response = requests.get(url=url, params=par)
    print('请求url:',response.url)
    if response.status_code != 200:
        # print('本次requests错误代码为: ',response.status_code)
        result=translateModelByNLPIR(text)
        # print('本次requests结果已经矫正为nlpir翻译引擎:',result)
        return result
    if '<html>' in response.text:
        # print('本次requests.<html>错误代码为: ',response.status_code)
        result=translateModelByNLPIR(text)
        # print('本次requests结果已经矫正为nlpir翻译引擎:',result)
        return result
    if'请输入翻译类型' in response.text:
        # print('本次requests.请输入翻译类型错误代码为: ',response.status_code)
        result=translateModelByNLPIR(text)
        # print('本次requests结果已经矫正为nlpir翻译引擎:',result)
        return result
    print(response.text)
    result = json.loads(response.text).get('text')
    print(result)
    # print('翻译结果为:',result)
    return result

def translationModelByYoudao(text):
    sentence = sent_tokenize(text)
    # print('本条文章英语句子分段展示：',sentence)
    # 翻译列表中的每个句子
    sum=''
    for item in sentence:
        print('===========================本条数据开始处理===========================')
        # print('替换空格前的每段句子:',item)
        item = item.replace('\n\n', '\n')
        # item = item.replace('\n', ' ')
        item = item.replace('\t\t', '\t')  # 替换换行符
        # print('替换空格后的每段句子:',item)  # 显示替换后的行
        sum=sum+translationModelByYD(item)
    # print('本条数据最终翻译结果为',sum)
    print('===========================本条数据结束处理===========================')
    return sum

if __name__ == '__main__':
    s="BAE Systems has successfully completed the delivery of the 1,000th F-35 Lightning II fuselage to Lockheed Martin"
    print(translationModelByYoudao(s))
    s="BAE delivers 1,000th F-35 Lightning II fuselage to Lockheed Martin in major milestone for the world’s largest defense programme Posted on February 9, 2023 by Seapower Staff Release from BAE Systems ******* BAE delivers 1,000th F-35 Lightning II fuselage to Lockheed Martin in major milestone for the world’s largest defense programme 7 Feb 2023 BAE Systems has delivered the 1,000th rear fuselage to Lockheed Martin for the F-35, the world’s most advanced and capable fifth generation fighter. More than 1,500 employees at the Company’s facilities in Samlesbury, Lancashire, produce the rear fuselage for every F-35 in the global fleet. The first fuselage was delivered to Lockheed Martin in 2005. Speech marks at an event today celebrating the 1000th delivery, Cliff Robson, Group Managing Director, BAE Systems Air, said: “This is a significant moment for everyone involved in the programme and a testament to the highly-skilled workforce we have in the North West of England. “Our role on the F-35 programme is another example of how we make a substantial contribution to the local and national UK economy and help to deliver capability which is critical for national security.” Speech marks Bridget Lauderdale, Lockheed Martin Vice President and General Manager of the F-35 programme, said: “The F-35 programme powers economic growth and prosperity for the UK injecting approximately £41billion* into the UK economy and supporting more than 20,000 jobs in the UK supply chain, many of those based in the North West. “With more than 500 companies in our UK supply chain, we’re proud of the role that our partnership with BAE Systems has in delivering the world’s most advanced aircraft for the UK and 17 other allied nations.” F-35 aircraft inside hangar BAE Systems has been involved in the F-35 programme since its inception and plays key roles across the development, manufacture and sustainment of the aircraft, which is operated by the Royal Air Force, Royal Navy and air forces across the world. The F-35s global programme of record amounts to more than 3,000 F-35s amongst the programme’s 17 customers. Work on the programme will continue at BAE Systems’ advanced manufacturing hub at Samlesbury for many years to come. Speech marks Susan Addison, Senior Vice President for US Programmes at BAE Systems Air, said: “This is an important milestone for our business and demonaates both the expertise of our people and their commitment to delivering for the F-35 programme. “The roles we play today are underpinned by a world-class manufacturing pedigree and induaial know-how in the UK, which has been developed through decades of cutting edge experience in combat air programmes. We are proud of what we do for our customers and the air forces who help keep us safe.”"
    print(translationModelByYoudao(s))