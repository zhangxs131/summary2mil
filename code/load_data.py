import json
import pandas as pd

def load_content_json(file_name):
    if file_name.split('.')[-1]=='csv':
        df=pd.read_csv(file_name)

    print(len(df))
    return df

def main():
    file_name='../data/content_1000.json'
    content=load_content_json(file_name)

if __name__=="__main__":
    main()