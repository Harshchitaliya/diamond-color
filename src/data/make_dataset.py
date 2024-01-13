import pandas as pd
import numpy as np
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import yaml

# load data
def load_dataset(datapath):
    df = pd.read_csv(datapath)
    return df

# split data into train-test
def data_split(df,test_split,seed):
    X=df.iloc[:,170:330]
    Y=df["color"]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_split, random_state=seed)

    le=LabelEncoder()
    le.fit(y_train)
    y_train=pd.DataFrame(le.transform(y_train))
    y_test=pd.DataFrame(le.transform(y_test))

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))

    return X_train, X_test, y_train, y_test

# save data
def save_data(X_train, X_test, y_train, y_test,output_path):
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_path + 'X_train.csv', index=False)
    X_test.to_csv(output_path + 'X_test.csv', index=False)
    y_train.to_csv(output_path + 'y_train.csv', index=False)
    y_test.to_csv(output_path + 'y_test.csv', index=False)


def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["make_dataset"]

    data_path = home_dir.as_posix() + "/data/build_data/df.csv"
    output_path = home_dir.as_posix() + '/data/processed/'

    df = load_dataset(data_path)
    X_train,X_test,y_train,y_test = data_split(df, params['test_split'], params['seed'])
    save_data(X_train,X_test,y_train,y_test, output_path)


if __name__ == "__main__":
    main()