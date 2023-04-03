import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_name='../中文情报.csv'):
    with open(file_name,'r',encoding='utf-8') as f:
        content=f.read().splitlines()

    result=[eval(i) for i in content]

    return result


def split_train_test(file_name='../中文情报.csv',test_size=0.05):
    res=load_data(file_name)
    train_list,test_list = train_test_split(res, test_size=test_size, random_state=42)

    gen_csv(train_list,'../train.csv')
    gen_csv(test_list,'../test.csv')


def gen_csv(result_list,save_name='train.csv'):
    my_dict = {}
    for item in result_list:
        for key, value in item.items():
            if key in my_dict:
                my_dict[key].append(value)
            else:
                my_dict[key] = [value]

    df=pd.DataFrame(data=my_dict)
    df.to_csv(save_name,index=None)


def show_data(file_name='../中文情报.csv'):

    result=load_data(file_name)

    print(len(result))


def main():

    split_train_test('../中文情报.csv')

if __name__=='__main__':
    main()
