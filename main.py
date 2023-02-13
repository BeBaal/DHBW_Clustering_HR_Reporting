"""This program is used  for an exploratory data analysis of
    different clustering methods regarding their use for reporting
    purposes. Data used is aggregated reporting data from Lidl regional
    departments and mostly on the topic Human Resources. The Analysis
    is part of my research project "Clusteringverfahren und deren
    Einsatzmöglichkeiten im Personalreporting: Ein Anwendungsbeispiel"
    at DHBW CAS in Heilbronn for my Master in Business Informatics.

    License: MIT
    Author: Bernd Baalmann
"""
# Import statements
import os
import time
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

# Options
OPTION_POWERPOINT_OR_WORD = 'Word'  # Word or Powerpoint
OPTION_DELETE_FORMER_RESULTS = True
OPTION_USE_STANDARDSCALER = True
OPTION_DESCALING_KEYFIGURES = True
OPTION_FILTER_COUNTRIES = False
# If OPTION_FILTER_COUNTRIES equals true filter to following countries:
OPTION_COUNTRY_LIST = [
    "DE",
    "FR",
    "US",
    "NL"
]
OPTION_FILTER_KEYFIGURES = False

# Attributes for Clustering
OPTION_FILTER_KEYFIGURES_LIST = [
    "Eintritte",
    "Austritte",
    "Austritte pro MA",
    "Eintritte Frühfluktuation",
    "Eintritte Frühfluktuation pro MA",
    "Austritte Frühfluktuation",
    "Austritte Frühfluktuation pro MA"
    "tats. AZV",
    "maximale AZV",
    "MA mit Zeiterfassung",
    "Vertragsstunden",
    "Bruttostunden",
    "Bruttostunden pro MA",
    "Krankstunden bez.",
    "Krankenstand bez. %",
    "Krankstunden tats.",
    "Krankstunden pro MA",
    "Resturlaub",
    # for export
]

# Attributes with high correlation
# OPTION_FILTER_KEYFIGURES_LIST = [
#     "Eintritte",
#     "Austritte",
#     "Eintritte Frühfluktuation",
#     "Eintritte Frühfluktuation pro MA",
#     "Austritte Frühfluktuation",
#     "maximale AZV",
#     "tats. AZV",
#     "Bruttostunden pro MA",
#     "Krankenstand bez. %",
#     "Krankstunden pro MA",
#     "Eintritte pro MA",
#     "Austritte pro MA",
#     "Austritte Frühfluktuation pro MA",
#     "Frühfluktuation",
#     "Durchschnittsalter",
#     "Fluktuation",
#     "tats. AZV pro MA mit Zeiterfassung",
#     "Vertragsstunden pro MA",
#     "Krankenstand",
#     "Resturlaub pro Kopf"
# ]


def main():
    """This is the main method of the program. Here the loop logic is
    implemented and most of the function calls"""
    logging.info("Main function was called")

    # get the start time
    start_time = time.time()

    delete_results()

    matplotlib_settings()

    dataframe = load_files()

    save_statistical_summary(dataframe)

    if OPTION_FILTER_COUNTRIES is True:
        dataframe = filter_countries(dataframe)

    if OPTION_FILTER_KEYFIGURES is True:
        dataframe = filter_keyfigures(dataframe)

    number_of_countries = len(dataframe["Lidl Land"].unique())

    # Generate information about the dataset
    plot_distribution(dataframe)
    plot_correlation(dataframe)

    # Set x variable for clustering that stays the same
    keyfigure_x = 'Fluktuation'

    # Get a list of keyfigures to set keyfigure_y to
    keyfigures = get_keyfigures(dataframe)

    # Loop over y keyfigures
    for keyfigure_y in keyfigures:
        # comparison with itself is not necessary
        if keyfigure_y == keyfigure_x:
            continue

        # Remove not necessary features
        data = setup_data_clustering_algorithm(dataframe,
                                               keyfigure_x,
                                               keyfigure_y)

        clustering(data,
                   2,
                   keyfigure_x,
                   keyfigure_y)

        clustering(data,
                   number_of_countries,
                   keyfigure_x,
                   keyfigure_y)

        plot_density(data,
                     keyfigure_x,
                     keyfigure_y)

        # Remove not necessary features
        data = setup_data_clustering_traditionally(dataframe,
                                                   keyfigure_x,
                                                   keyfigure_y)

        traditional_clustering(data,
                               keyfigure_x,
                               keyfigure_y)

    # get end time
    end_time = time.time()

    # calculate calculation time
    elapsed_time = end_time - start_time
    print('Execution time:', elapsed_time, 'seconds')


def matplotlib_settings():
    """This function sets the matplotlib global export settings either for
    Powerpoint or Word relating to the option that was set in the class
    variables.
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "12"

    match OPTION_POWERPOINT_OR_WORD:
        case 'Word':
            plt.rcParams["figure.figsize"] = (15, 10)
        case 'Powerpoint':
            plt.rcParams["figure.figsize"] = (20, 20)


def descale_dataframe(dataframe, scaler):
    """This function descales the data with the mean and standard
    deviation of the former used standard scaler object.

    Args:
        dataframe (numpy array): HR KPI dataframe
        scaler (standard scaler object): Standard Scaler

    Returns:
        numpy array: HR KPI numpy array
    """
    if OPTION_DESCALING_KEYFIGURES is True:
        # transforms Dataframe to numpy array
        data = scaler.inverse_transform(dataframe)

    return data


def scale_dataframe(dataframe):
    """This function scales the data with the standard scaler. Sets
    mean to 0 and the standard deviation to 1. In the scaler object the
    former mean and standard deviation is saved for future descaling.

    Args:
        dataframe (pandas dataframe): HR KPI dataframe

    Returns:
        numpy array: HR KPI dataframe
        standard scaler object: Standard Scaler
    """

    if OPTION_USE_STANDARDSCALER is True:
        # Scale the values from 0 to 1
        scaler = StandardScaler(copy=False)
        # transforms Dataframe to numpy array
        data = scaler.fit_transform(dataframe)
        scaler.set_output(transform="pandas")

    return data, scaler


def delete_results():
    """For convenience and error pruning this function deletes the old results
    files from the result folders of the clustering figures. The descriptive
    summary does not get deleted.
    """

    if OPTION_DELETE_FORMER_RESULTS is False:
        return

    paths = [r'C:\FPA2\Figures\BIRCH\\',
             r'C:\FPA2\Figures\DBScan\\',
             r'C:\FPA2\Figures\Gaussian\\',
             r'C:\FPA2\Figures\KMeans\\',
             r'C:\FPA2\Figures\Traditional_Clusters\\',
             r'C:\FPA2\Figures\Agglomeratives_Clustering\\',
             r'C:\FPA2\Figures\Density\\']

    for path in paths:
        for file_name in os.listdir(path):

            # construct full file path
            file = path + file_name
            if os.path.isfile(file):
                logging.info('Deleting file:', file)
                os.remove(file)


def filter_countries(dataframe):
    """This function takes the class variable OPTION_COUNTRIES and
    filters the dataframe correspondingly.

    Args:
        dataframe (pandas dataframe): HR KPI dataframe

    Returns:
        pandas dataframe: Filtered HR KPI dataframe
    """

    dataframe = dataframe.loc[dataframe['Lidl Land'].isin(OPTION_COUNTRY_LIST)]

    return dataframe


def plot_density(data, keyfigure_x, keyfigure_y):
    """This method plots the density of two keyfigures from a dataframe.

    Args:
        dataframe (pandas dataframe): HR KPI dataframe
        input_x (string): First keyfigure
        input_y (string): Second keyfigure
    """
    logging.info('plot_2_keyfigures_categorical function was called')

    filenpath_and_name = str(r'C:\FPA2\Figures\Density\Plot_'
                             + keyfigure_y
                             + '.svg')

    # set seaborn style
    sns.set_style("white")

    # Basic 2D density plot
    sns.kdeplot(data=data,
                x=keyfigure_x,
                y=keyfigure_y)

    plt.savefig(filenpath_and_name,
                bbox_inches="tight")

    plt.close()


def traditional_clustering(dataframe, keyfigure_x, keyfigure_y):
    """This function defines the traditional reporting cluster
    categories and does the loop logic over the different clusters.
    Additionally the corresponding plotting function is called here.

    Args:
        dataframe (pandas dataframe): HR KPI dataframe
        keyfigure_x (string): First keyfigure for analysis
        keyfigure_y (string): Second keyfigure for analysis
    """
    categories = ["Lidl Land", "Lidl Gesellschaftstyp"]

    for category in categories:
        plot_2_keyfigures_categorical(
            dataframe,
            category,
            keyfigure_x,
            keyfigure_y)


def filter_keyfigures(dataframe):
    """_summary_

    Args:
        dataframe (_type_): _description_
    """
    if OPTION_FILTER_KEYFIGURES is True:
        for keyfigure in OPTION_FILTER_KEYFIGURES_LIST:
            dataframe = dataframe.drop(columns=keyfigure)

    return dataframe


def get_keyfigures(dataframe):
    """This functions checks the dataframe for relevant keyfigures and
    gives them back as a list. Also removes unnecessary features.

    Args:
        dataframe (pandas dataframe): HR KPI dataframe

    Returns:
        list: HR keyfigures for analysis from dataset
    """
    # Get names of columns
    keyfigures = list(dataframe)

    # Remove not necessary features
    keyfigures.remove('Concat')
    keyfigures.remove('Lidl Land')
    keyfigures.remove('Lidl Land Langtext')
    keyfigures.remove('Lidl Gesellschaftstyp')
    keyfigures.remove('Lidl Gesellschaften')

    return keyfigures


def load_files():
    """Loads Excel files and returns a dataframe

    Returns:
        pandas dataframe: Sheet Daten_GES from excel  file as a dataframe
    """

    logging.info("Load_files function was called")

    xls = pd.ExcelFile(
        r'C:\FPA2\Daten_Forschungsprojektarbeit_2\Daten_FPA2.xlsx')
    data_file = pd.read_excel(xls, 'Export', header=1)

    # Load dataframes
    dataframe = pd.DataFrame(data_file)

    return dataframe


def setup_data_clustering_algorithm(dataframe, keyfigure_x, keyfigure_y):
    """This method removes unnecessary parts from the dataframe, deletes
    Null Values, establishes a standard scaling and sets the result up
    as a numpy array.

    Args:
        dataframe (pandas dataframe): HR KPI dataframe
        keyfigure_x (string): First keyfigure for analysis
        keyfigure_y (string): Second keyfigure for analysis

    Returns:
        numpy array: data for further analysis
    """
    logging.info('Setup function was called')

    #  Remove unnecessary features from dataframe
    dataframe = dataframe.loc[:, [
        keyfigure_x,
        keyfigure_y]]

    # data cleansing
    # KMeans can not handle NaN Values and zero values are not relevant
    dataframe.replace(0, np.nan, inplace=True)
    dataframe.dropna(axis=0, how="any", inplace=True)

    return dataframe


def setup_data_clustering_traditionally(dataframe, keyfigure_x, keyfigure_y):
    """This method removes unnecessary parts from the dataframe and
    deletes Null Values.

    Args:
        dataframe (pandas dataframe): HR KPI dataframe
        keyfigure_x (string): First keyfigure for analysis
        keyfigure_y (string): Second keyfigure for analysis

    Returns:
        pandas dataframe: data for further analysis
    """
    logging.info('Setup function was called')

    #  Remove unnecessary features from dataframe
    dataframe = dataframe.loc[:,
                              ["Lidl Land",
                               "Lidl Gesellschaftstyp",
                               keyfigure_x,
                               keyfigure_y]]

    # data cleansing
    # No NaN and zero values
    dataframe.replace(0, np.nan, inplace=True)
    dataframe.dropna(axis=0, how="any", inplace=True)

    return dataframe


def plot_2_keyfigures_categorical(
        dataframe,
        category,
        keyfigure_x,
        keyfigure_y):
    """This method plots two keyfigures from a dataframe categorically.

    Args:
        dataframe (pandas dataframe): HR KPI dataframe
        input_x (string): First keyfigure
        input_y (string): Second keyfigure
    """
    logging.info('plot_2_keyfigures_categorical function was called')

    filenpath_and_name = str(r'C:\FPA2\Figures\Traditional_Clusters\Plot_'
                             + category
                             + "_"
                             + keyfigure_y
                             + '.svg')

    match category:
        case "Gesellschaftstyp":
            figure_legend_columns = 1  # only two entries therefore one column
        case "Lidl Land":
            figure_legend_columns = 2  # lots of possible entries two columns
        case _:
            figure_legend_columns = 1  # fallback solution

    sns.scatterplot(data=dataframe,
                    x=keyfigure_x,
                    y=keyfigure_y,
                    hue=category)

    plt.title(category + " " + keyfigure_x + " / " + keyfigure_y)
    plt.xlabel(keyfigure_x)
    plt.ylabel(keyfigure_y)
    plt.legend(ncol=figure_legend_columns,
               bbox_to_anchor=(1.04, 1),
               loc="upper left")
    plt.savefig(filenpath_and_name,
                bbox_inches="tight")
    plt.close()


def plot_distribution(dataframe):
    """This method plots the distribution of the relevant keyfigures.

    Args:
        dataframe (pandas dataframe): HR KPI dataframe
    """
    logging.info('plot_distribution function was called')

    filenpath_and_name = r'C:\FPA2\Figures\Attribute_Distribution.svg'

    dataframe.hist(bins=25,
                   figsize=(40, 40),
                   color='b',
                   alpha=0.6
                   )

    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.savefig(filenpath_and_name, bbox_inches="tight")
    plt.close()


def plot_correlation(dataframe):
    """This method plots the correlation of the relevant keyfigures. See also
    https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e
    for example heatmap implementation.

    Args:
        dataframe (pandas dataframe): HR KPI dataframe
    """
    logging.info('plot_cross_correlation function was called')

    # Triangle cross correlation matrix
    filenpath_and_name = r'C:\FPA2\Figures\Attribute_Cross_Correlation.svg'

    mask = np.triu(np.ones_like(dataframe.corr(numeric_only=True),
                                dtype=np.bool_))

    heatmap = sns.heatmap(dataframe.corr(numeric_only=True),
                          vmin=-1,
                          vmax=1,
                          mask=mask,
                          annot=True,
                          cmap='BrBG',
                          fmt=".1f")

    heatmap.set_title('Correlation Heatmap')

    plt.tight_layout()
    plt.savefig(filenpath_and_name)
    plt.close()

    # Single correlation
    filenpath_and_name = r'C:\FPA2\Figures\Attribute_Single_Correlation.svg'

    dataframe.corr(numeric_only=True)[['Fluktuation']].sort_values(
        by='Fluktuation',
        ascending=False)

    heatmap = sns.heatmap(dataframe.corr(numeric_only=True)[
        ['Fluktuation']].sort_values(
        by='Fluktuation',
        ascending=False),
        vmin=-1,
        vmax=1,
        annot=True,
        cmap='BrBG')

    heatmap.set_title('Features Correlating with Fluctuation',
                      fontdict={'fontsize': 10},
                      pad=16)

    plt.tight_layout()
    plt.savefig(filenpath_and_name)
    plt.close()


def save_statistical_summary(dataframe):
    """This method saves a descriptive statistical summary for a
    overview of the dataset.

    Args:
        dataframe (pandas dataframe):  HR KPI dataframe
    """
    logging.info('save_statistical_summary method was called')

    # Statistical key figures for the dataset
    statistical_data = dataframe.describe(include='all')
    statistical_data.to_excel(
        r'C:\FPA2\Daten_Forschungsprojektarbeit_2\StatisticalData.xlsx')


def clustering(data, number_of_clusters, keyfigure_x, keyfigure_y):
    """This method calls the clustering methods and is acting as an
    interface to the different clustering algorithms.

    Args:
        data (numpy array): Bivariate HR keyfigures for regional departments
        keyfigure_x (string): First keyfigure
        keyfigure_y (string): Second keyfigure
    """
    logging.info('clustering method was called')

    # centroid based clustering methods
    kmeans(data,
           number_of_clusters,
           keyfigure_x,
           keyfigure_y)

    # Density based clustering methods
    dbscan(data,
           keyfigure_x,
           keyfigure_y)
    # optics

    # distribution based clustering methods
    gaussian(data,
             number_of_clusters,
             keyfigure_x,
             keyfigure_y)

    # Hierarchy based clustering methods
    birch(data,
          number_of_clusters,
          keyfigure_x,
          keyfigure_y)

    agglomerative_clustering(data,
                             number_of_clusters,
                             keyfigure_x,
                             keyfigure_y)
    # mean-shift


def kmeans(dataframe, number_cluster, keyfigure_x, keyfigure_y):
    """This method clusters the data via k means.

    Args:
        data (numpy array): Two dimensional array of HR keyfigures
        number_cluster (integer): Number of clusters for analysis
        keyfigure_x (string): First keyfigure
        keyfigure_y (string): Second keyfigure
    """
    logging.info('clustering method kmeans was called')

    filenpath_and_name = str(r'C:\FPA2\Figures\KMeans\Plot_C'
                             + str(number_cluster)
                             + "_"
                             + keyfigure_y
                             + '.svg')

    if number_cluster <= 2:
        figure_legend_columns = 1  # only two entries therefore one columns
    else:
        figure_legend_columns = 2  # lots of entries therefore two columns

    dataframe, scaler = scale_dataframe(dataframe)

    # Declaring Model with some parameters
    model = KMeans(
        n_clusters=number_cluster,
        n_init='auto'
    )

    # Fitting Model
    model.fit(dataframe)

    # predict the labels of clusters.
    label = model.fit_predict(dataframe)

    # Getting the centroids center
    centroids = model.cluster_centers_

    # Getting unique labels
    u_labels = np.unique(label)

    centroids = descale_dataframe(centroids, scaler)

    dataframe = descale_dataframe(dataframe, scaler)

    # plotting the results
    for i in u_labels:
        plt.scatter(dataframe[label == i, 0],
                    dataframe[label == i, 1],
                    label='Cluster ' + str(i) + ' n=' + str(
                        np.count_nonzero(label == i)))

    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                c='black')

    plt.title("KMEANS C_" + str(number_cluster) +
              " " + keyfigure_x + " / " + keyfigure_y)
    plt.xlabel(keyfigure_x)
    plt.ylabel(keyfigure_y)
    plt.legend(ncol=figure_legend_columns,
               bbox_to_anchor=(1.04, 1),
               loc="upper left")

    plt.savefig(filenpath_and_name,
                bbox_inches="tight")
    plt.close()


def gaussian(dataframe, number_cluster, keyfigure_x, keyfigure_y):
    """This method clusters the data via k means.

    Args:
        data (numpy array): Bivariate HR keyfigures for regional departments
        number_cluster (integer): Number of clusters for analysis
        keyfigure_x (string): First keyfigure
        keyfigure_y (string): Second keyfigure
    """
    logging.info('clustering method kmeans was called')

    filenpath_and_name = str(r'C:\FPA2\Figures\Gaussian\Plot_C'
                             + str(number_cluster)
                             + "_"
                             + keyfigure_y
                             + '.svg')

    if number_cluster <= 2:
        figure_legend_columns = 1  # only two entries therefore one columns
    else:
        figure_legend_columns = 2  # lots of entries therefore two columns

    dataframe, scaler = scale_dataframe(dataframe)

    # Declaring Model
    model = GaussianMixture(n_components=number_cluster)

    # Fitting Model
    model.fit(dataframe)

    # predict the labels of clusters.
    label = model.fit_predict(dataframe)

    # Getting unique labels
    u_labels = np.unique(label)

    dataframe = descale_dataframe(dataframe, scaler)

    # plotting the results
    for i in u_labels:
        plt.scatter(dataframe[label == i, 0],
                    dataframe[label == i, 1],
                    label='Cluster ' + str(i) + ' n=' + str(
                        np.count_nonzero(label == i)))

    plt.title("Gaussian C_" + str(number_cluster) + " " +
              keyfigure_x + " / " + keyfigure_y)
    plt.xlabel(keyfigure_x)
    plt.ylabel(keyfigure_y)
    plt.legend(ncol=figure_legend_columns,
               bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.savefig(filenpath_and_name, bbox_inches="tight")
    plt.close()


def dbscan(dataframe, keyfigure_x, keyfigure_y):
    """This method clusters the data via dbscan.

    Args:
        data (numpy array): Bivariate HR keyfigures for regional departments
        keyfigure_x (string): First keyfigure for analysis
        keyfigure_y (string): Second keyfigure for analysis
    """
    logging.info('clustering method dbscan was called')

    filenpath_and_name = str(r'C:\FPA2\Figures\DBscan\Plot_'
                             + keyfigure_y
                             + '.svg')

    dataframe, scaler = scale_dataframe(dataframe)

    # Declaring Model
    model = DBSCAN()

    # Fitting Model
    model.fit(dataframe)

    # predict the labels of clusters.
    label = model.fit_predict(dataframe)

    # Getting unique labels
    u_labels = np.unique(label)

    dataframe = descale_dataframe(dataframe, scaler)

    # plotting the results
    for i in u_labels:
        plt.scatter(dataframe[label == i, 0],
                    dataframe[label == i, 1],
                    label='Cluster ' + str(i) + ' n=' + str(
                        np.count_nonzero(label == i)))

    plt.title("DBSCAN " + keyfigure_x + " / " + keyfigure_y)
    plt.xlabel(keyfigure_x)
    plt.ylabel(keyfigure_y)
    plt.legend(ncol=1, bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(filenpath_and_name, bbox_inches="tight")
    plt.close()


def birch(dataframe, number_cluster, keyfigure_x, keyfigure_y):
    """This method clusters the data via birch.

    Args:
        data (numpy array): Bivariate HR keyfigures for regional departments
        number_cluster (integer): Number of clusters for analysis
        keyfigure_x (string): First keyfigure
        keyfigure_y (string): Second keyfigure
    """
    logging.info('clustering method kmeans was called')

    filenpath_and_name = str(r'C:\FPA2\Figures\BIRCH\Plot_C'
                             + str(number_cluster)
                             + "_"
                             + keyfigure_y
                             + '.svg')

    if number_cluster <= 2:
        figure_legend_columns = 1  # only two entries therefore one columns
    else:
        figure_legend_columns = 2  # lots of entries therefore two columns

    dataframe, scaler = scale_dataframe(dataframe)

    # Declaring Model with some parameters
    model = Birch(
        n_clusters=number_cluster
    )

    # Fitting Model
    model.fit(dataframe)

    # predict the labels of clusters.
    label = model.fit_predict(dataframe)

    # Getting unique labels
    u_labels = np.unique(label)

    dataframe = descale_dataframe(dataframe, scaler)

    # plotting the results
    for i in u_labels:
        plt.scatter(dataframe[label == i, 0],
                    dataframe[label == i, 1],
                    label='Cluster ' + str(i) + ' n=' + str(
                        np.count_nonzero(label == i)))

    plt.title("Birch C_"
              + str(number_cluster)
              + " "
              + keyfigure_x
              + " / "
              + keyfigure_y)
    plt.xlabel(keyfigure_x)
    plt.ylabel(keyfigure_y)
    plt.legend(ncol=figure_legend_columns,
               bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(filenpath_and_name, bbox_inches="tight")
    plt.close()


def agglomerative_clustering(
        dataframe,
        number_cluster,
        keyfigure_x,
        keyfigure_y):
    """This method clusters the data via agglomerative_clustering

    Args:
        data (numpy array): Bivariate HR keyfigures for regional departments
        number_cluster (integer): Number of clusters for analysis
        keyfigure_x (string): First keyfigure
        keyfigure_y (string): Last keyfigure
    """
    logging.info('clustering method agglomerative_clustering was called')

    filenpath_and_name = str(r"C:\FPA2\Figures\Agglomeratives_Clustering\Plot"
                             + "_Cluster"
                             + str(number_cluster)
                             + "_"
                             + keyfigure_y
                             + '.svg')

    if number_cluster <= 2:
        figure_legend_columns = 1  # only two entries therefore one columns
    else:
        figure_legend_columns = 2  # lots of entries therefore two columns

    dataframe, scaler = scale_dataframe(dataframe)

    # Declaring Model with some parameters
    model = AgglomerativeClustering(
        n_clusters=number_cluster
    )

    # Fitting Model
    model.fit(dataframe)

    # predict the labels of clusters.
    label = model.fit_predict(dataframe)

    # Getting unique labels
    u_labels = np.unique(label)

    dataframe = descale_dataframe(dataframe, scaler)

    # plotting the results
    for i in u_labels:
        plt.scatter(dataframe[label == i, 0],
                    dataframe[label == i, 1],
                    label='Cluster ' + str(i) + ' n=' + str(
                        np.count_nonzero(label == i)))

    plt.title("Agglomeratives Clustering C_"
              + str(number_cluster)
              + " "
              + keyfigure_x
              + " / "
              + keyfigure_y)
    plt.xlabel(keyfigure_x)
    plt.ylabel(keyfigure_y)
    plt.legend(ncol=figure_legend_columns,
               bbox_to_anchor=(1.04, 1),
               loc="upper left")
    plt.savefig(filenpath_and_name,
                bbox_inches="tight")
    plt.close()


main()
