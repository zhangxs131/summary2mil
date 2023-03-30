import json
import requests
from

def translate(sent):

    try:
        head = {"Content-Type": "application/json; charset=UTF-8"}
        url = 'http://182.92.69.3:7017/niutrans/textTranslation?apikey=3197af0f21041b4eff7e105eb4b1fd94'
        data = {"from": "en", "to": "zh", "src_text": sent, "termDictionaryLibraryId": '', "realmCode":"0",
               "translationMemoryLibraryId": ''}
        data = json.dumps(data)
        data = data.encode("utf-8")
        response = requests.post(url, headers=head, data=data)
        result = json.loads(response.text)
        # print(result)
        resultcn = []
        for datas in result['data']:
            for dt in datas['sentences']:
                resultcn.append(dt['data'])
            zhdata = ''.join(resultcn)
        return zhdata
    except:
        return ''


a= translate('apple')
print(a)
print(translate("Russian Aircraft manufacturer, UAC began deliveries of the latest multifunctional Su-30SM2 fighters to the Russian Armed Forces as part of the state defense order.\nDubbed \"Super Sukhoi,\" the Su-30SM2 is a modernized version of the Su-30SM. The first aircraft of the new modification in the naval livery have already departed from the Irkutsk aviation plant to the place of its deployment, the Russian Ministry of Defense (MOD)’s TV channel, tvzvezda.ru reported today.\nThe Su-30SM2’s main areas of improvement over the Su-30SM is an advanced engine and a new phased array radar, which dramatically expands its combat capabilities, according to the report. At the same time, the fighter retained all the advantages of the basic version: supermanoeuvrability, long flight range and a wide arsenal of weapons.\nThe Su-30SM fighter was tested with the AL-41F-1S only a couple of months ago. It is part a Russian MoD’s plan of achieving as much commonality with the Su-35 as possible.\nThe new engine for the Su-30SM2 is the AL-41F-1S TVC engine derived from the top-of-the-line Su-35 jet. The new engine will provide an increased thrust-to-weight ratio, lower fuel consumption and longer time between overhauls when compared with the outgoing Al-31FP engine.\nAlso new in the Su-30SM2 is PESA radar, the N011M Bars-R derivative with increased detection and tracking performance (compared to the radar on the Su-30SM). The SM2 upgrade includes the OSNOD multi-channel communication and information distribution system which enables the aircraft’s integration in Russia’s new-generation command-and-control network.\n"))