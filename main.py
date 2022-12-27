"""_summary_
"""
# Import statements
import logging

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def main():
    """This is the main method of the program """

    logging.info("Main function was called")

    dataframe = load_files()

    # plot_distribution(dataframe)
    # save_statistical_summary(dataframe)

    # Set initial figures for clustering
    keyfigure_x = 'Fluktuation'  # stays the same

    keyfigures = get_keyfigures(dataframe)

    # plot_2_attributes(dataframe
    # keyfigure_x,
    # keyfigure_y))

    for keyfigure_y in keyfigures:
        # comparison with itself not necessary
        if keyfigure_y == 'Fluktuation':
            continue

        # Remove not necessary attributes and scale data
        data = setup_data(
            dataframe,
            keyfigure_x,
            keyfigure_y)

        clustering(data, keyfigure_x, keyfigure_y)


def get_keyfigures(dataframe):
    """_summary_

    Args:
        dataframe (_type_): _description_

    Returns:
        list: List of keyfigures for analysis
    """
    # Get names of columns
    keyfigures = list(dataframe)

    # Remove not necessary attributes
    keyfigures.remove('Concat')
    keyfigures.remove('Lidl Land')
    keyfigures.remove('Lidl Land Langtext')
    keyfigures.remove('Lidl Gesellschaftstyp')
    keyfigures.remove('Lidl Gesellschaften')

    return keyfigures


def load_files():
    """Loads Excel files and returns a dataframe

    Returns:
        dataframe object: Sheet Daten_GES as a dataframe
    """

    logging.info("Load_files function was called")

    xls = pd.ExcelFile(
        r'C:\FPA2\Daten_Forschungsprojektarbeit_2\Daten_FPA2.xlsx')
    data_file = pd.read_excel(xls,  'Export', header=1)

    # Load dataframes
    dataframe = pd.DataFrame(data_file)

    return dataframe


def setup_data(dataframe, keyfigure_x, keyfigure_y):
    """ This method loads the data and sets it up"""
    logging.info('Setup function was called')

    #  Remove unnecessary attributes from dataframe
    dataframe = dataframe.loc[:, [
        keyfigure_x,
        keyfigure_y]]

    # data cleansing
    # KMeans can not handle NaN Values
    dataframe.dropna(axis=0, how="any", inplace=True)

    # Scale the values from 0 to 1
    scaler = StandardScaler(copy=False)
    # transforms Dataframe to numpy array
    data = scaler.fit_transform(dataframe)

    return data


def plot_2_attributes(dataframe, input_x, input_y):
    """ This method plots the results """
    logging.info('show_results function was called')

    sns.scatterplot(data=dataframe,
                    x=input_x,
                    y=input_y,
                    hue='Lidl Land')
    plt.show()


def plot_distribution(dataframe):
    """ This method plots the results """
    logging.info('plot_distribution function was called')

    dataframe.hist(bins=30, figsize=(20, 20), color='b', alpha=0.6)
    plt.show()


def save_statistical_summary(dataframe):
    """ This method saves the results """
    logging.info('save_statistical_summary method was called')

    # Statistical key figures for the dataset
    statistical_data = dataframe.describe(include='all')
    statistical_data.to_excel(
        r'C:\FPA2\Daten_Forschungsprojektarbeit_2\StatisticalData.xlsx')


def clustering(data, keyfigure_x, keyfigure_y):
    """ This method calls the clustering methods """
    logging.info('clustering method was called')

    kmeans(data, 3, keyfigure_x, keyfigure_y)


def kmeans(data, number_cluster, keyfigure_x, keyfigure_y):
    """ This method clusters the data via k means """
    logging.info('clustering method kmeans was called')

    filenpath_and_name = r'C:\FPA2\Figures\KMeans\Plot_' + keyfigure_y + '.png'

    # Declaring Model with some parameters
    model = KMeans(
        init="random",
        n_clusters=number_cluster,
        n_init=10,
        max_iter=300,
        random_state=42
    )

    # Fitting Model
    model.fit(data)

    # predict the labels of clusters.
    label = model.fit_predict(data)

    # Getting the centroids center
    centroids = model.cluster_centers_

    # Getting unique labels
    u_labels = np.unique(label)

    # plotting the results
    for i in u_labels:
        plt.scatter(data[label == i, 0],
                    data[label == i, 1],
                    label='Cluster ' + str(i))

    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                c='black')

    plt.title(keyfigure_x + " / " + keyfigure_y)
    plt.xlabel(keyfigure_x)
    plt.ylabel(keyfigure_y)
    plt.text(60, .025, 'n=15')
    plt.legend()

    plt.savefig(filenpath_and_name)
    plt.show()


main()
