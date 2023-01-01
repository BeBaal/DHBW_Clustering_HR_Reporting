"""This program is used  for an exploratory data analysis of different clustering
    methods regarding their use for reporting purposes. Data used is aggregated
    reporting data from Lidl regional departments and mostly on the topic
    Human Resources. The Analysis is part of my research project at DHBW CAS in
    Heilbronn for my Master in Business Informatics.

    Author: Bernd Baalmann
"""
# Import statements
import logging
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture


def main():
    """This is the main method of the program """
    logging.info("Main function was called")

    dataframe = load_files()

    plot_distribution(dataframe)

    save_statistical_summary(dataframe)

    # Set initial figures for clustering
    keyfigure_x = 'Fluktuation'  # stays the same

    keyfigures = get_keyfigures(dataframe)

    for keyfigure_y in keyfigures:
        # comparison with itself not necessary
        if keyfigure_y == keyfigure_x:
            continue

        # Remove not necessary attributes and scale data
        data = setup_data(
            dataframe,
            keyfigure_x,
            keyfigure_y)

        clustering(data, keyfigure_x, keyfigure_y)


def get_keyfigures(dataframe):
    """This functions checks the dataframe for relevant keyfigures and gives
    them back as a list. Also removes unnecessary attributes.

    Args:
        dataframe (pandas dataframe): dataframe of the analysis

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
        pandas dataframe: Sheet Daten_GES as a dataframe
    """

    logging.info("Load_files function was called")

    xls = pd.ExcelFile(
        r'C:\FPA2\Daten_Forschungsprojektarbeit_2\Daten_FPA2.xlsx')
    data_file = pd.read_excel(xls,  'Export', header=1)

    # Load dataframes
    dataframe = pd.DataFrame(data_file)

    return dataframe


def setup_data(dataframe, keyfigure_x, keyfigure_y):
    """This method removes unnecessary parts from the dataframe, deletes Null
    Values, establishes a standard scaling and sets the result up as a numpy
    array.

    Args:
        dataframe (pandas dataframe): HR KPI dataframe
        keyfigure_x (string): First keyfigure for analysis
        keyfigure_y (string): Second keyfigure for analysis

    Returns:
        numpy array: data for further analysis
    """
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
    """This method plots two attributes from a dataframe.

    Args:
        dataframe (pandas dataframe): HR KPI dataframe
        input_x (string): First keyfigure
        input_y (string): Second keyfigure
    """
    logging.info('plot_2_attributes function was called')

    sns.scatterplot(data=dataframe,
                    x=input_x,
                    y=input_y,
                    hue='Lidl Land')
    plt.show()


def plot_distribution(dataframe):
    """This method plots the distribution of the relevant keyfigures.

    Args:
        dataframe (pandas dataframe): HR KPI dataframe
    """
    logging.info('plot_distribution function was called')

    filenpath_and_name = r'C:\FPA2\Figures\Attribute_Distribution.png'

    dataframe.hist(bins=30,
                   figsize=(20, 20),
                   color='b',
                   alpha=0.6)
    plt.savefig(filenpath_and_name)
    plt.close()


def save_statistical_summary(dataframe):
    """This method saves a descriptive statistical summary for a overview of
    the dataset.

    Args:
        dataframe (pandas dataframe):  HR KPI dataframe
    """
    logging.info('save_statistical_summary method was called')

    # Statistical key figures for the dataset
    statistical_data = dataframe.describe(include='all')
    statistical_data.to_excel(
        r'C:\FPA2\Daten_Forschungsprojektarbeit_2\StatisticalData.xlsx')


def clustering(data, keyfigure_x, keyfigure_y):
    """This method calls the clustering methods and is acting as an interface
    to the different clustering algorithms.

    Args:
        data (numpy array): Two dimensional array of HR keyfigures for regional departments
        keyfigure_x (string): First keyfigure
        keyfigure_y (string): Second keyfigure
    """
    logging.info('clustering method was called')

    # centroid based clustering methods
    kmeans(data, 3, keyfigure_x, keyfigure_y)

    # Density based clustering methods
    dbscan(data, keyfigure_x, keyfigure_y)
    # optics

    # distribution based clustering methods
    gaussian(data, keyfigure_x, keyfigure_y)

    # Hierarchy based clustering methods
    birch(data, 3, keyfigure_x, keyfigure_y)
    agglomerative_clustering(data, 3, keyfigure_x, keyfigure_y)
    # mean-shift


def kmeans(data, number_cluster, keyfigure_x, keyfigure_y):
    """This method clusters the data via k means.

    Args:
        data (numpy array): Two dimensional array of HR keyfigures for regional departments
        number_cluster (integer): Number of clusters for analysis
        keyfigure_x (_type_): First keyfigure
        keyfigure_y (_type_): Second keyfigure
    """
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
                    label='Cluster ' + str(i) + ' n='+str(np.count_nonzero(label == i)))

    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                c='black')

    plt.title("KMEANS "+keyfigure_x + " / " + keyfigure_y)
    plt.xlabel(keyfigure_x)
    plt.ylabel(keyfigure_y)
    plt.text(60, .025, 'n=15')
    plt.legend()

    plt.savefig(filenpath_and_name)
    plt.close()


def gaussian(data, keyfigure_x, keyfigure_y):
    pass


def dbscan(data, keyfigure_x, keyfigure_y):
    """This method clusters the data via dbscan.

    Args:
        data (numpy array): Two dimensional array of HR keyfigures for regional departments
        keyfigure_x (_type_): First keyfigure for analysis
        keyfigure_y (_type_): Second keyfigure for analysis
    """
    logging.info('clustering method dbscan was called')

    filenpath_and_name = r'C:\FPA2\Figures\DBscan\Plot_' + keyfigure_y + '.png'

    # Declaring Model
    model = DBSCAN()

    # Fitting Model
    model.fit(data)

    # predict the labels of clusters.
    label = model.fit_predict(data)

    # Getting unique labels
    u_labels = np.unique(label)

    # plotting the results
    for i in u_labels:
        plt.scatter(data[label == i, 0],
                    data[label == i, 1],
                    label='Cluster ' + str(i) + ' n='+str(np.count_nonzero(label == i)))

    plt.title("DBSCAN "+keyfigure_x + " / " + keyfigure_y)
    plt.xlabel(keyfigure_x)
    plt.ylabel(keyfigure_y)
    plt.text(60, .025, 'n=15')
    plt.legend()

    plt.savefig(filenpath_and_name)
    plt.close()


def birch(data, number_cluster, keyfigure_x, keyfigure_y):
    """This method clusters the data via birch.

    Args:
        data (numpy array): Two dimensional array of HR keyfigures for regional departments
        number_cluster (integer): Number of clusters for analysis
        keyfigure_x (string): First keyfigure
        keyfigure_y (string): Second keyfigure
    """
    logging.info('clustering method kmeans was called')

    filenpath_and_name = r'C:\FPA2\Figures\BIRCH\Plot_' + keyfigure_y + '.png'

    # Declaring Model with some parameters
    model = Birch(
        n_clusters=number_cluster
    )

    # Fitting Model
    model.fit(data)

    # predict the labels of clusters.
    label = model.fit_predict(data)

    # Getting unique labels
    u_labels = np.unique(label)

    # plotting the results
    for i in u_labels:
        plt.scatter(data[label == i, 0],
                    data[label == i, 1],
                    label='Cluster ' + str(i) + ' n='+str(np.count_nonzero(label == i)))

    plt.title("Birch "+keyfigure_x + " / " + keyfigure_y)
    plt.xlabel(keyfigure_x)
    plt.ylabel(keyfigure_y)
    plt.legend()
    plt.savefig(filenpath_and_name)
    plt.close()


def agglomerative_clustering(data, number_cluster, keyfigure_x, keyfigure_y):
    """This method clusters the data via agglomerative_clustering

    Args:
        data (numpy array): Two dimensional array of HR keyfigures for regional departments
        number_cluster (integer): Number of clusters for analysis
        keyfigure_x (string): First keyfigure
        keyfigure_y (string): Last keyfigure
    """
    logging.info('clustering method agglomerative_clustering was called')

    filenpath_and_name = r'C:\FPA2\Figures\Agglomeratives_Clustering\Plot_' + \
        keyfigure_y + '.png'

    # Declaring Model with some parameters
    model = AgglomerativeClustering(
        n_clusters=number_cluster
    )

    # Fitting Model
    model.fit(data)

    # predict the labels of clusters.
    label = model.fit_predict(data)

    # Getting unique labels
    u_labels = np.unique(label)

    # plotting the results
    for i in u_labels:
        plt.scatter(data[label == i, 0],
                    data[label == i, 1],
                    label='Cluster ' + str(i) + ' n='+str(np.count_nonzero(label == i)))

    plt.title("Agglomeratives Clustering "+keyfigure_x + " / " + keyfigure_y)
    plt.xlabel(keyfigure_x)
    plt.ylabel(keyfigure_y)
    plt.legend()
    plt.savefig(filenpath_and_name)
    plt.close()


main()
