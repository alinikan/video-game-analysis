import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

from cleaning import clean_data
from analyze_origin import analyze_origin_data
from cleaning_playtime import clean_playtime_data

sns.set_palette('pastel')
plt.style.use('ggplot')


def main():
    # Load the dataset
    df_all = pd.read_csv("data/steam_games.csv")

    # Perform analysis on the original data
    df_original = df_all.copy()
    analyze_origin_data(df_original)

    # Data cleaning and generate cleaned data file
    clean_data(df_all)

    df_cleaned = pd.read_csv("data/cleaned_steam_games.csv")

    # df2_column_names = ["name", "type", "time", "0"]
    # df2 = pd.read_csv("data/steam-200k.csv", names=df2_column_names)
    # df2_cleaned = clean_playtime_data(df2)

if __name__ == "__main__":
    main()
