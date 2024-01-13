import pandas as pd
import numpy as np
import pathlib


def build_feature(datafile):

    df = pd.read_csv(datafile)

    # split the first column of the dataframe
    color = df["0"].str.split("_").str[2]
    type = df["0"].str.split("_").str[3]
    light = df["0"].str.split("_").str[0]


    # make new columns
    df["color"] = color
    df["type"] = type
    df["light"] = light

    # select dataframe which light value is only W
    df = df[df["light"] == "W"]

    # Select a dataframe which type column value is none
    df = df[df["type"]=="None"]

    # outlier detect and remove
    def detect_outliers(column):
        # Calculate the mean and standard deviation
        mean = column.mean()
        std = column.std()

        # Calculate the z-scores
        z_scores = (column - mean) / std

        # Identify outliers
        outliers = column[(z_scores < -3) | (z_scores > 3)]

        return outliers
    
        # Detect outliers in each column
    for column in df.columns:
        if df[column].dtype != 'object':
            outliers = detect_outliers(df[column])

            # Remove outliers
            df.drop(outliers.index, inplace=True)


    #choose color from dataframe
    list = ["D","E","F","G","H","I","J","K"]
    df = df[df["color"].isin(list)] 

    return df


def save_data(df,output_path):
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path  + "df.csv",index = False)



def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    datafile = home_dir.as_posix() + "/data/raw/dfcolor.csv"
    output_path = home_dir.as_posix() + "/data/build_data/"

    df = build_feature(datafile)
    save_data(df,output_path)


if __name__ == "__main__":
    main()






    