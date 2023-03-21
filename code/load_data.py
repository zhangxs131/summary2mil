import json
import pandas as pd

def load_content_json(file_name):
    if file_name.split('.')[-1]=='csv':
        df=pd.read_csv(file_name)
        print(len(df))
        for id, data in df.iterrows():
            print(data)
            break

        return df

    elif file_name.split('.')[-1]=='json':
        df_json=json.load(open(file_name,'r',encoding='utf-8'))
        print(len(df_json))
        print(df_json[-1])


        return df_json

def main():
    file_name='../data/整编数据.json'
    content=load_content_json(file_name)

if __name__=="__main__":
    main()