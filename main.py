"""_summary_
"""
# Import statements
import logging

import matplotlib.pyplot as plt
import pandas as pd


def main():
    """ This is the main method of the programm """
    print('Hello World')

    # Load Excel files
    xls = pd.ExcelFile(
        '../Daten_Forschungsprojektarbeit_2/ \
        Daten_Studienarbeit_Optimierung_Kapitalertragssteuer.xlsx')
    df_daten_ges = pd.read_excel(xls,  'Daten_GES', header=0)

    # Load Dataframes
    d_daten_ges = pd.DataFrame(df_daten_ges, columns=[
        'Lidl Land',
        'Lidl Gesellschaftsty',
        'Lidl Gesellschaften',
        'Anzahl Sätze',
        'Mitarbeiter',
        'Eintritte_',
        'Mitarbeiter Austritte_',
        'Mitarbeiter Fluktuation in %_',
        'Eintritte Frühflukt._',
        'Austritte Frühflukt._',
        'Frühflukt in %_',
        'Anzahl Mitarbeiter',
        'Mitarbeiter - Ø-Alter',
        'Krankenstand',
        'Arbeitszeitverstöße',
        'Resturlaub'
    ])

    print(d_daten_ges)


def setup():
    """ This method loads the data and sets it up"""
    logging.info('Setup function was called')


def show_results():
    """ This method plots the results """


def save_results():
    """ This function saves the results """
    logging.info('save_results function was called')
