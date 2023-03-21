# -*- coding: utf-8 -*-
# 控制台：https://ai.youdao.com/console/#/app-overview/check-application?appId=22f94ea69ee3aed9
# 文档：https://ai.youdao.com/DOCSIRMA/html/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E7%BF%BB%E8%AF%91/API%E6%96%87%E6%A1%A3/%E6%96%87%E6%9C%AC%E7%BF%BB%E8%AF%91%E6%9C%8D%E5%8A%A1/%E6%96%87%E6%9C%AC%E7%BF%BB%E8%AF%91%E6%9C%8D%E5%8A%A1-API%E6%96%87%E6%A1%A3.html#section-9
import requests
import json
from nltk import sent_tokenize    #以符号形式实现分句

def translationModelByYD(query):

    # 创建 session 对象
    session = requests.Session()

    # 设置请求头，模拟浏览器的访问
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Referer': 'https://www.google.com/',
    }

    # 设置 session 的请求头
    session.headers.update(headers)

    # # 发送第一次请求
    # response1 = session.get('https://example.com')
    #
    # # 发送第二次请求
    # response2 = session.get('https://example.com/another-page')
    #
    # # 打印响应内容
    # print(response1.content)
    # print(response2.content)

    url = 'http://fanyi.youdao.com/translate'
    data = {
        "i": query,  # 待翻译的字符串
        "from": "en",
        "to": "zh-CHS",
        "smartresult": "dict",
        "client": "fanyideskweb",
        "salt": "16081210430989",
        "doctype": "json",
        "version": "2.1",
        "keyfrom": "fanyi.web",
        "action": "FY_BY_CLICKBUTTION"
    }
    response = session.post(url, data=data)
    # print(response.text)
    # print(response.status_code)
    res = response
    #print('本次翻译消耗时间:',res.elapsed)
    if res.status_code!=200:
        print('本次requests错误代码为: ',res.status_code)
        return query
    # print('result is ', json.dump(response.text,ensure_ascii=False))
    # result=response.json().get(['translateResult'][0][0]['tgt'])
    print(response.text)
    #result = json.loads(response.text).get('text')
    result = response.json()
    #print(result['translateResult'][0][0]['tgt'])
    return result['translateResult'][0][0]['tgt']

def translationModelByYoudao(text):
    sentence = sent_tokenize(text)
    print('text lenght is ',text.__len__())
    # print('sentence is ',sentence)
    # 翻译列表中的每个句子
    result=''
    for item in sentence:
        result=result+translationModelByYD(item)
    return result
if __name__ == '__main__':
    s="BAE Systems has successfully completed the delivery of the 1,000th F-35 Lightning II fuselage to Lockheed Martin"
    print(translationModelByYD(s))
    s = "BAE delivers 1,000th F-35 Lightning II fuselage to Lockheed Martin in major milestone for the world’s largest defense programme Posted on February 9, 2023 by Seapower Staff Release from BAE Systems ******* BAE delivers 1,000th F-35 Lightning II fuselage to Lockheed Martin in major milestone for the world’s largest defense programme 7 Feb 2023 BAE Systems has delivered the 1,000th rear fuselage to Lockheed Martin for the F-35, the world’s most advanced and capable fifth generation fighter. More than 1,500 employees at the Company’s facilities in Samlesbury, Lancashire, produce the rear fuselage for every F-35 in the global fleet. The first fuselage was delivered to Lockheed Martin in 2005. Speech marks at an event today celebrating the 1000th delivery, Cliff Robson, Group Managing Director, BAE Systems Air, said: “This is a significant moment for everyone involved in the programme and a testament to the highly-skilled workforce we have in the North West of England. “Our role on the F-35 programme is another example of how we make a substantial contribution to the local and national UK economy and help to deliver capability which is critical for national security.” Speech marks Bridget Lauderdale, Lockheed Martin Vice President and General Manager of the F-35 programme, said: “The F-35 programme powers economic growth and prosperity for the UK injecting approximately £41billion* into the UK economy and supporting more than 20,000 jobs in the UK supply chain, many of those based in the North West. “With more than 500 companies in our UK supply chain, we’re proud of the role that our partnership with BAE Systems has in delivering the world’s most advanced aircraft for the UK and 17 other allied nations.” F-35 aircraft inside hangar BAE Systems has been involved in the F-35 programme since its inception and plays key roles across the development, manufacture and sustainment of the aircraft, which is operated by the Royal Air Force, Royal Navy and air forces across the world. The F-35s global programme of record amounts to more than 3,000 F-35s amongst the programme’s 17 customers. Work on the programme will continue at BAE Systems’ advanced manufacturing hub at Samlesbury for many years to come. Speech marks Susan Addison, Senior Vice President for US Programmes at BAE Systems Air, said: “This is an important milestone for our business and demonaates both the expertise of our people and their commitment to delivering for the F-35 programme. “The roles we play today are underpinned by a world-class manufacturing pedigree and induaial know-how in the UK, which has been developed through decades of cutting edge experience in combat air programmes. We are proud of what we do for our customers and the air forces who help keep us safe.”"

    print(translationModelByYoudao(s))