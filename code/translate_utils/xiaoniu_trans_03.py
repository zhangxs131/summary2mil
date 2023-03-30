
import json
import requests

def translate(sent):

    try:
        head = {"Content-Type": "application/json; charset=UTF-8"}
        url = 'http://182.92.69.3:7017/niutrans/textTranslation?apikey=3197af0f21041b4eff7e105eb4b1fd94'
        data = {"from": "en", "to": "zh", "src_text": sent, "termDictionaryLibraryId": '',
                "translationMemoryLibraryId": ''}
        data = json.dumps(data)
        data = data.encode("utf-8")
        response = requests.post(url, headers=head, data=data)
        result = json.loads(response.text)
        # print(result)
        resultcn = []
        print(result)
        for datas in result['data']:
            for dt in datas['sentences']:
                resultcn.append(dt['data'])
            zhdata = ''.join(resultcn)
        return zhdata
    except:
        return ''


print(translate('the apple for me '))
a= translate('apple')
print(a)