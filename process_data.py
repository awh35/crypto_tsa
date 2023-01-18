"""Script to format data, adding a column for pair name, and setting the 
index as the appropriate datetime."""

from glob import glob
from tqdm import tqdm
import pandas as pd


def get_pair_name(file_name):
    return file_name.replace("data/", "").replace(".csv", "")


def process_csvs():
    files = glob("data/*")
    for file in tqdm(files):
        df = pd.read_csv(file)
        df["pair_name"] = get_pair_name(file)
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df["date"] = df["time"].dt.date
        df.to_csv(file, index=False)

    
if __name__ == "__main__":
    process_csvs()