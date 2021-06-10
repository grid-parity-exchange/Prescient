import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir("./downloads")

output_data = pd.read_csv("collated_output_1.csv")


def produce_hist(col_name):
    plt.figure()
    plt.hist(output_data[col_name])
    plt.xlabel(col_name)
    plt.ylabel("Frequency")
    plt.title("Histogram of " + col_name + " over Fictitious Scenarios")
    plt.show()


def produce_scatter(col1, col2):
    plt.figure()
    plt.scatter(output_data[col1], output_data[col2])
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(col1 + " vs " + col2)
    plt.show()