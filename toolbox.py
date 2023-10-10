import pandas as pd
import numpy as np
import scipy
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.cbook import boxplot_stats
import missingno as msno
import seaborn as sns
import textwrap
import re


def display_distribution_empirique(dataframe: pd.DataFrame, categorie: str, plot: bool):
    """
    Affiche un tableau représentant la distribution empirique d'une variable donnée et 
    optionnellement un histogramme

    Positional arguments : 
    -------------------------------------
    dataframe : pd.DataFrame : jeu de données contenant les indicateurs 
    categorie : str : catégorie d'indicateurs que l'on souhaite analyser
    plot : bool : afficher ou non la distribution empirique sous forme d'histogramme
    """
    plt.rcParams["figure.figsize"] = [14, 7]

    effectifs = dataframe[categorie].value_counts()
    distribution_empirique = pd.DataFrame(effectifs.index, columns=[categorie])
    distribution_empirique['n'] = effectifs.values
    distribution_empirique['f (%)'] = round(
        distribution_empirique['n']/len(dataframe) * 100, 2)

    display(distribution_empirique)

    if plot:
        sns.set_theme(style='whitegrid', palette='Set2')
        sns.barplot(data=distribution_empirique, x=categorie, y='n')
        plt.xticks(rotation=90)
        plt.title('Distribution Empirique ' + categorie, fontsize='25',
                  color='#737373', fontname='Arial Rounded MT Bold', pad='10')
        plt.ylabel('Nombre d\'indicateurs par ' + categorie,
                   color='#737373', fontsize='20', fontname='Arial Rounded MT Bold')
        plt.show()


def get_subset_by_topic(original_dataset: pd.DataFrame, topic_name: str, visualization=True):
    """
    Retourne un sous-échantillon du jeu de données filtré sur un thème donné

    Positional arguments : 
    -------------------------------------
    original_dataset : pd.DataFrame : jeu de données initial à filtrer
    topic_name : str : thème d'indicateurs que l'on souhaite analyser

    Optionnal arguments : 
    -------------------------------------
    visualization : bool : afficher ou non la distribution empirique du jeu de données filtré.
    """

    mask_topic = original_dataset['Topic'] == topic_name
    subset_by_topic = original_dataset.loc[mask_topic, :].copy()

    if visualization:
        list_sources = subset_by_topic['Source'].unique()
        if len(list_sources) == 1:
            print('Topic ' + topic_name, len(list_sources), "seule source:\n")
        else:
            print('Topic ' + topic_name, len(list_sources),
                  "sources différentes:\n")
        print(*list_sources, sep="\n")
        display_distribution_empirique(subset_by_topic, 'Source', False)

    return subset_by_topic


def find_substring(mainstring: str, regex):
    """
    Retourne un tuple indiquant la présence ou non de la chaine de caractère recherchée 
    et la chaîne de caractère elle même 

    Positional arguments : 
    -------------------------------------
    mainstring : str : chaîne de caractères recherchée
    regex : : Objet regex dans lequel rechercher la chaîne de caractères
    """
    match = regex.search(mainstring)
    if match:
        return (True, match[0])
    return (False, "")


def drop_gender(dataset: pd.DataFrame, column_name: str):
    """
    Retourne un jeu de données (DataFrame) sans les indicateurs calculés par genre 

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial, à nettoyer
    column_name : str : nom de la colonne contenant les noms des indicateurs
    """
    regex_gender = re.compile(r'((\s(f|F)e|\s)(m|M)(ale))|(\sgender parity)')
    mask_gender = ~dataset[column_name].apply(
        lambda x: find_substring(x, regex_gender)[0])
    subset = dataset.loc[mask_gender, :].copy()

    return subset


def drop_quintile(dataset: pd.DataFrame, column_name: str):
    """
    Retourne un jeu de données (DataFrame) sans les indicateurs calculés par 

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial, à nettoyer
    column_name : str : nom de la colonne contenant les noms des indicateurs
    """
    regex_quintile = re.compile(r'((Q|q)uintile)')
    mask_quintile = ~dataset[column_name].apply(
        lambda x: find_substring(x, regex_quintile)[0])
    subset = dataset.loc[mask_quintile, :].copy()

    return subset


def drop_area(dataset: pd.DataFrame, column_name: str):
    """
    Retourne un jeu de données (DataFrame) sans les indicateurs calculés par zone d'habitation 

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial, à nettoyer
    column_name : str : nom de la colonne contenant les noms des indicateurs
    """
    regex_area = re.compile(r'((R|r)ural)|((U|u)rban)')
    mask_area = ~dataset[column_name].apply(
        lambda x: find_substring(x, regex_area)[0])
    subset = dataset.loc[mask_area, :].copy()

    return subset


def remove_age(dataset: pd.DataFrame, column_name: str):
    """
    Retourne un jeu de données en séparant l'âge et le nom de l'indicateur

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial, à nettoyer
    column_name : str : nom de la colonne contenant les noms des indicateurs
    """
    subset = dataset.copy()
    regex_age = re.compile(r'\s(a|A)(ge\s)(\d{1,2})(\-\d{2}|\+)')
    subset['Age'] = subset[column_name].apply(
        lambda x: find_substring(x, regex_age)[1])
    subset[column_name + ' without age'] = subset.apply(
        lambda row: row[column_name].replace(row['Age'], ""), axis=1)

    return subset


def drop_education_details(dataset: pd.DataFrame, column_name: str):
    """
    Retourne un jeu de données (DataFrame) sans les indicateurs calculés pour un type d'établissement 
    particulier ou par type d'études réalisées

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial, à nettoyer
    column_name : str : nom de la colonne contenant les noms des indicateurs
    """
    list_filter = ['lower secondary', 'general', 'vocational', 'private institution', 'public institution',
                   'ISCED', 'Education programmes', 'Humanities and Arts', 'Social Sciences, Business and Law', 'Science',
                   'Engineering, Manufacturing and Construction', 'Agriculture', 'Health and Welfare', 'Services']

    mask_education = ~dataset[column_name].str.contains(
        '|'.join(list_filter), case=False)
    subset = dataset.loc[mask_education, :].copy()

    return subset


def get_columns_year(dataset: pd.DataFrame):
    """
    Retourne une liste contenant les noms des colonnes correspondant à des années

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial dont on va extraire les années
    """
    mask_isyear = dataset.columns.str.contains('^[0-9]{4}', regex=True)
    columns_isyear = dataset.columns[mask_isyear].values

    return columns_isyear


def keep_columns_year_and(dataset: pd.DataFrame, columns_to_keep: list[str]):
    """
    Retourne un sous-échantillon du jeu de données (dataframe) contenant les 
    colonnes années et une liste de colonnes spécifiée

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial à filtrer
    columns_to_keep : list of str : noms des colonnes à garder
    """
    columns_isyear = get_columns_year(dataset)
    columns_to_keep.extend(columns_isyear)

    subset = dataset[columns_to_keep].copy()

    return subset


def drop_region(dataset: pd.DataFrame, column_name: str):
    """
    Retourne un sous-échantillon du jeu de données (dataframe) après supression des 
    lignes correspondant à une région et non à un pays

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial à filtrer
    column_name : str : nom de la colonne contenant les pays/régions
    """
    list_region = ['Arab World', 'East Asia & Pacific', 'East Asia & Pacific (excluding high income)', 'Euro area',
                   'Europe & Central Asia', 'Europe & Central Asia (excluding high income)', 'European Union',
                   'Heavily indebted poor countries (HIPC)', 'High income', 'Latin America & Caribbean',
                   'Latin America & Caribbean (excluding high income)', 'Least developed countries: UN classification',
                   'Low & middle income', 'Low income', 'Lower middle income', 'Middle East & North Africa',
                   'Middle East & North Africa (excluding high income)', 'Middle income', 'North America', 'OECD members',
                   'South Asia', 'Sub-Saharan Africa', 'Sub-Saharan Africa (excluding high income)', 'Upper middle income',
                   'World']

    mask_region = ~dataset[column_name].isin(list_region)
    subset = dataset.loc[mask_region].copy()

    return subset


def drop_indicator(dataset: pd.DataFrame, indicator_list: list[str], regex=False):
    """
    Retourne un sous-échantillon du jeu de données (dataframe) après supression 
    d'une liste d'indicateurs donnés

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial à filtrer
    indicator_list : list of str : liste des indicateurs à supprimer du tableau

    Optionnal arguments : 
    -------------------------------------
    regex : bool : indique si la liste d'indicateurs contient 
    des expressions régulières ou non
    """
    if regex:
        mask_indicator = ~dataset['Indicator Name'].str.contains(
            '|'.join(indicator_list), case=False)
    else:
        mask_indicator = ~dataset['Indicator Name'].isin(indicator_list)

    subset = dataset.loc[mask_indicator, :].copy()

    indicators_before_drop = len(dataset['Indicator Name'].unique())
    indicators_after_drop = len(subset['Indicator Name'].unique())

    print(indicators_before_drop - indicators_after_drop,
          "indicateur(s) supprimé(s)")
    print('Il reste', indicators_after_drop, 'indicateurs')

    return subset


def drop_country(dataset: pd.DataFrame, country_list: list[str]):
    """
    Retourne un sous-échantillon du jeu de données (dataframe) après supression 
    d'une liste de pays donnés

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial à filtrer
    country_list : list of str : liste des pays à retirer du tableau
    """
    mask_country = ~dataset['Country Name'].isin(country_list)
    subset = dataset.loc[mask_country, :].copy()

    countries_after_drop = len(subset['Country Name'].unique())
    droppped_countries = dataset.loc[~mask_country, 'Country Name'].unique()

    print(len(droppped_countries), "pays supprimé(s)")
    print('Il reste', countries_after_drop, 'pays\n')

    if len(droppped_countries) > 0:
        print(pd.DataFrame({'Dropped countries': droppped_countries}).head(
            len(droppped_countries)), '\n')

    return subset


def drop_years_before(dataset: pd.DataFrame, before_year: int):
    """
    Retourne un sous-échantillon du jeu de données (dataframe) après supression 
    des colonnes correspondant à des années avant une année donnée.

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial à filtrer
    before_year : int : année de référence
    """
    columns_year = get_columns_year(dataset).astype(int)
    filter_year = filter(lambda year: year < before_year, columns_year)
    list_years = [str(year) for year in list(filter_year)]

    list_columns = dataset.columns[dataset.columns.isin(list_years)].values
    subset = dataset.drop(columns=list_columns)

    return subset


def drop_years_after(dataset: pd.DataFrame, after_year: int):
    """
    Retourne un sous-échantillon du jeu de données (dataframe) après supression 
    des colonnes correspondant à des années après une année donnée.

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial à filtrer
    after_year : int : année de référence
    """
    columns_year = get_columns_year(dataset).astype(int)
    filter_year = filter(lambda year: year > after_year, columns_year)
    list_years = [str(year) for year in list(filter_year)]

    list_columns = dataset.columns[dataset.columns.isin(list_years)].values

    subset = dataset.drop(columns=list_columns)

    return subset


def missing_values_by(dataset: pd.DataFrame, column_category: str):
    """
    Retourne un dataframe répertoriant le nombre de valeurs manquantes par indicateur ou par pays

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial à analyser
    column_category : str : indique si l'on veut analyser les valeurs manquantes par pays ou par indicateur
    """
    list_values = []
    for name in dataset[column_category].unique():
        mask_name = dataset[column_category] == name
        columns_isyear = get_columns_year(dataset)

        subset = dataset.loc[mask_name, columns_isyear].copy()
        list_values.append([name, subset.isnull().sum().sum()])

    missing_values_df = pd.DataFrame(
        list_values, columns=[column_category, 'Number of Missing Values'])
    missing_values_df['Missing Values (%)'] = round(
        missing_values_df['Number of Missing Values'] / (subset.shape[0] * subset.shape[1]) * 100, 2)
    missing_values_df = missing_values_df.set_index(column_category)
    missing_values_df = missing_values_df.sort_values(
        'Number of Missing Values', ascending=False)

    return missing_values_df


def missing_values_by_year(dataset: pd.DataFrame):
    """
    Retourne un dataframe répertoriant le nombre de valeurs manquantes par année

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial à analyser
    """
    columns_isyear = get_columns_year(dataset)
    subset = dataset[columns_isyear]

    missing_values_serie = subset.isnull().sum()
    missing_values_df = missing_values_serie.to_frame(
        name='Number of Missing Values')
    missing_values_df = missing_values_df.reset_index().rename(columns={
        'index': 'Year'})

    missing_values_df['Missing Values (%)'] = round(
        missing_values_df['Number of Missing Values'] / (dataset.shape[0]) * 100, 2)

    missing_values_df = missing_values_df.sort_values(
        'Number of Missing Values', ascending=False)

    return missing_values_df


def get_values_to_drop(dataset: pd.DataFrame, column_name: str, threshold: int):
    """
    Retourne une liste d'indicateurs ou de pays à supprimer du jeu de données 
    car leur pourcentage de valeurs manquantes est supérieur à une limite donnée

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial à analyser
    column_name : str : nom de la colonne contenant les noms d'indicateurs ou de pays
    threshold : int : pourcentage limite de valeurs manquantes accepté
    """
    missing_values_df = missing_values_by(dataset, column_name).reset_index()
    mask_threshold = missing_values_df['Missing Values (%)'] >= threshold
    values_to_drop = missing_values_df.loc[mask_threshold, column_name]

    return values_to_drop


def drop_indicators_by_threshold(dataset: pd.DataFrame, threshold: int):
    """
    Retourne un sous-échantillon du jeu de données (dataframe) après suppression des indicateurs avec un
    pourcentage de valeurs manquantes supérieur à une limite donnée

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial à filtrer
    threshold : int : pourcentage limite de valeurs manquantes accepté
    """
    indicators_to_drop = get_values_to_drop(
        dataset, 'Indicator Name', threshold)
    subset = drop_indicator(dataset, indicators_to_drop.values)

    return subset


def drop_countries_by_threshold(dataset: pd.DataFrame, threshold: int):
    """
    Retourne un sous-échantillon du jeu de données (dataframe) après suppression des pays avec un
    pourcentage de valeurs manquantes supérieur à une limite donnée

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial à filtrer
    threshold : int : pourcentage limite de valeurs manquantes accepté
    """
    countries_to_drop = get_values_to_drop(dataset, 'Country Name', threshold)
    subset = drop_country(dataset, countries_to_drop.values)

    return subset


def drop_years_by_threshold(dataset: pd.DataFrame, threshold: float):
    """
    Retourne un sous-échantillon du jeu de données (dataframe) après suppression des années avec un
    pourcentage de valeurs manquantes supérieur à une limite donnée

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial à filtrer
    threshold : float : pourcentage limite de valeurs manquantes accepté
    """
    subset = dataset.dropna(axis=1, thresh=(1-threshold) * dataset.shape[0])

    nb_years_before_drop = len(get_columns_year(dataset))
    nb_years_after_drop = len(get_columns_year(subset))

    print(nb_years_before_drop-nb_years_after_drop,
          "année(s) supprimée(s).\nIl reste", nb_years_after_drop, "années.")

    return subset


def get_subset_direction_indicateur_country(dataset: pd.DataFrame):
    """
    Retourne un sous-échantillon du jeu de données (dataframe) avec une ligne par année 
    et une colonne par pays (contenant le nombre d'indicateurs renseignés par pays).

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial à transformer
    """
    subset = keep_columns_year_and(dataset, ['Country Name', 'Indicator Name'])
    columns_isyear = get_columns_year(subset)
    subset['Count'] = subset[columns_isyear].count(1)
    subset.drop(columns=columns_isyear, inplace=True)

    subset = subset.pivot_table(
        index='Country Name', columns='Indicator Name', values='Count')
    subset.replace(0, np.nan, inplace=True)

    return subset


def get_subset_direction_indicateur_year(dataset: pd.DataFrame):
    """
    Retourne un sous-échantillon du jeu de données (dataframe) 
    avec une ligne par année et une colonne par indicateur (contenant le nombre de pays renseignés par indicateur)

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial à transformer
    """
    subset = keep_columns_year_and(dataset, ['Indicator Name'])
    subset = subset.groupby('Indicator Name').count()
    subset.replace(0, np.nan, inplace=True)
    subset = subset.transpose()

    return subset


def get_first_year_not_null(row):
    """
    Retourne un entier correspondant à la première année contenant une valeur non nulle

    Positional arguments : 
    -------------------------------------
    row :  : ligne d'un dataframe
    """
    mask_isyear = row.index.str.contains('^[0-9]{4}', regex=True)
    list_year_notnull = row.loc[(mask_isyear) & (row.values != 0)].index.values

    return list_year_notnull[0]


def get_last_year_not_null(row):
    """
    Retourne un entier correspondant à la dernière année contenant une valeur non nulle

    Positional arguments : 
    -------------------------------------
    row :  : ligne d'un dataframe
    """
    mask_isyear = row.index.str.contains('^[0-9]{4}', regex=True)
    list_year_notnull = row.loc[(mask_isyear) & (row.values != 0)].index.values

    return list_year_notnull[-1]


def get_first_last_year_with_value(dataset: pd.DataFrame):
    """
    Retourne un dataframe avec une ligne par indicateur, une colonne
    contenant la première année avec une valeur non nulle 
    et une colonne contenant la dernière année avec une valeur non nulle

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données à filtrer
    """
    columns_to_drop = ['Country Name', 'Country Code', 'Indicator Code']
    subset = dataset.drop(columns=columns_to_drop).copy()

    subset = subset.groupby('Indicator Name').count()
    subset['first year with value'] = subset.apply(
        lambda row: get_first_year_not_null(row), axis=1)
    subset['last year with value'] = subset.apply(
        lambda row: get_last_year_not_null(row), axis=1)

    subset = subset[['first year with value', 'last year with value']]

    display_distribution_empirique(subset, 'first year with value', False)
    display_distribution_empirique(subset, 'last year with value', False)

    return subset


def plot_missing_values_indicators(dataset: pd.DataFrame, palette_name: str, color_position: int, on: str, all_str: str):
    """
    Affiche un graphique permettant de visualiser le nombre de valeurs manquantes par indicateur

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données à analyser
    palette_name : str : palette de couleurs à utiliser 
    color_position : int : emplacement de la couleur à utiliser dans la palette 
    on : str : label de l'axe des ordonnées 
    all_str : str : titre du graphique
    """
    color_list_plot = sns.color_palette(palette_name, 9)
    rgb_plot = color_list_plot[color_position]

    color_list_text = sns.color_palette('Greys', 15)
    rgb_text = color_list_text[11]

    title_text = '\nMissing Values for selected indicators (by ' + \
        all_str + ')'

    with plt.style.context('seaborn-white'):
        plt.rcParams['xtick.major.pad'] = '15'
        mat = msno.matrix(dataset, color=rgb_plot)
        plt.title(title_text, fontsize='35', color=rgb_text,
                  fontname='Arial Rounded MT Bold', pad=45)
        plt.ylabel('Number of ' + on, color=rgb_text,
                   fontsize='20', fontname='Arial Rounded MT Bold')

        labels_list = [textwrap.fill(label.get_text(), 18)
                       for label in mat.axes.get_xticklabels()]
        plt.xticks([0, 1, 2, 3, 4], labels=labels_list, ha='center', color=rgb_text,
                   fontname='Arial Rounded MT Bold', fontsize='19', rotation=0)

        plt.show()


def get_max_consecutive_missing_values(row):
    """
    Retourne un entier, le nombre maximum de valeurs manquantes consécutives dans une ligne donnée d'un tableau

    Positional arguments : 
    -------------------------------------
    row : : ligne d'un dataframe
    """
    mask_isyear = row.index.str.contains('^[0-9]{4}', regex=True)
    row = row.loc[mask_isyear]

    row_isnull = row.isnull()
    row_isnull_cumsum = row_isnull.cumsum()
    consecutive_null = row_isnull_cumsum.sub(
        row_isnull_cumsum.mask(row_isnull).ffill().fillna(0)).astype(int)

    consecutive_null_max = consecutive_null.max()

    return consecutive_null_max


def drop_countries_with_consecutive_null(dataset: pd.DataFrame, subset_indicator: pd.DataFrame):
    """
    Retourne un échantillon d'un jeu de données (dataframe) après suppression 
    des pays avec plus de valeurs manquantes que le quantile d'ordre 0.9 pour un indicateur donné

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données à filtrer
    subset_indicator : pd.DataFrame : jeu de données filtré sur un indicateur
    """
    subset = keep_columns_year_and(subset_indicator, ['Country Name'])
    subset['Max Consecutive Missing Values'] = subset.apply(
        lambda row: get_max_consecutive_missing_values(row), axis=1)

    nb_consecutive_missing_values = subset['Max Consecutive Missing Values'].sort_values(
    ).values
    thresh = np.quantile(nb_consecutive_missing_values, .9)
    if thresh == 0.0:
        print("Le quantile d'ordre 0.9 est :", thresh,
              ".\nOn ne supprime que les pays vides")
        thresh = len(get_columns_year(subset))
    else:
        print("Le quantile d'ordre 0.9 est :", thresh,
              ".\nOn supprime donc tous les pays avec plus de", thresh, "valeurs manquantes consécutives.")

    countries_to_drop = subset.loc[subset['Max Consecutive Missing Values']
                                   >= thresh, 'Country Name'].unique()
    new_dataset = drop_country(dataset, countries_to_drop)

    return new_dataset


def interpolate(row):
    """
    Retourne une ligne de dataframe après imputation par interpolation linéaire

    Positional arguments : 
    -------------------------------------
    row :  : ligne de dataframe
    """
    mask_isyear = row.index.str.contains('^[0-9]{4}', regex=True)
    row.loc[mask_isyear] = row.loc[mask_isyear].astype(
        float).interpolate(method='linear', limit_direction='both')

    return row


def plot_missing_values(dataset: pd.DataFrame, stage: str, palette_name: str, color_position: int):
    """
    Affiche un graphique permettant de visualiser les valeurs manquantes

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données à analyser
    stage : str : "avant" ou "après" interpolation linéaire
    palette_name : str : nom de la palette de couleurs seaborn à utiliser
    color_position : int : emplacement de la couleur à utiliser dans la palette
    """
    color_list_plot = sns.color_palette(palette_name, 9)
    rgb_plot = color_list_plot[color_position]

    color_list_text = sns.color_palette('Greys', 15)
    rgb_text = color_list_text[11]

    title_text = '\nMissing Values for indicator \"' + \
        dataset['Indicator Name'].unique()[0] + '\"\n-' + \
        stage + ' interpolation -'

    with plt.style.context('seaborn-white'):
        msno.matrix(keep_columns_year_and(
            dataset, ['Country Name']), color=rgb_plot)
        plt.title(title_text, fontsize='30', color=rgb_text,
                  fontname='Arial Rounded MT Bold', pad=20)
        plt.ylabel('Number of Countries', color=rgb_text,
                   fontsize='20', fontname='Arial Rounded MT Bold')
        plt.xticks(color=rgb_text,
                   fontname='Arial Rounded MT Bold', fontsize='18')
        plt.show()


def filter_outlier(dataset: pd.DataFrame, x_column: str):
    """
    Retourne un échantillon d'un jeu de données (dataframe) après suppression des valeurs extrêmes

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données à filtrer
    x_column : str : nom de la colonne sur laquelle filtrer les valeurs extrêmes
    """
    outliers = [y for stat in boxplot_stats(
        dataset[x_column]) for y in stat['fliers']]
    mask_outliers = dataset[x_column].isin(outliers)
    subset = dataset.loc[~mask_outliers]

    return subset


def reshape_dataset_forboxplot(dataset: pd.DataFrame, column_to_count: str):
    """
    Retourne un dataframe transformé pour faciliter la création de boxplots

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données à filtrer
    column_to_count : str : nom de la colonne dont on veut afficher les occurences dans le boxplots
    """
    effectif = dataset.groupby([column_to_count])[
        ['Country Name']].nunique().to_dict()['Country Name']

    subset = dataset.copy()
    subset[column_to_count +
           ' n'] = subset.apply(lambda row: effectif[row[column_to_count]], axis=1)
    subset[column_to_count + ' (n)'] = subset.apply(lambda row: row[column_to_count] +
                                                    '\n(n=' + str(row[column_to_count + ' n']) + ')', axis=1)

    return subset


def plot_boxplot_by_indicator(dataset: pd.DataFrame, x_column: str, y_column: str, with_scatter: bool):
    """
    Affiche un boxplot par indicateur

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant les valeurs à utiliser dans les boxplots
    x_column : str : nom de la colonne contenant les valeurs à mettre en abscisse du boxplot
    y_column : str : nom de la colonne contenant les valeurs à mettre en ordonnée du boxplot
    with_scatters : bool : laisser ou non les valeurs individuelles apparentes
    """
    color_list_text = sns.color_palette('Greys', 15)
    rgb_text = color_list_text[12]

    dico_param = {}
    if with_scatter:
        dico_param = {'palette': 'bright', 'title': 'ScatterBoxplot', 'markerfacecolor': '#01FBEE', 'color': '#00145A',
                      'boxprops': {'edgecolor': '#00145A', 'linewidth': 4.0, 'facecolor': 'none'}}
    else:
        dico_param = {'palette': 'Set2', 'title': 'Boxplot', 'markerfacecolor': 'coral', 'color': 'black',
                      'boxprops': {'edgecolor': 'black', 'linewidth': 4.0}}

    with plt.style.context('seaborn-white'):
        plt.rcParams['axes.labelpad'] = '40'
        sns.set_theme(style='whitegrid', palette=dico_param['palette'])
        fig, axes = plt.subplots(3, 2, figsize=(40, 60), sharey=True)
        fig.tight_layout()
        suptitle_text = dico_param['title'] + ' by ' + y_column.replace(
            '(n)', '') + '(' + x_column + ')\n - for each indicator - '
        fig.suptitle(suptitle_text, fontname='Arial Rounded MT Bold',
                     fontsize=60, color=rgb_text)
        plt.subplots_adjust(left=None, bottom=None, right=None,
                            top=0.91, wspace=0.1, hspace=0.5)
        (l, c) = (0, 0)

    for indicator in relevant_indicators:
        mask_indicator = dataset['Indicator Name'] == indicator
        subset_indicator = dataset.loc[mask_indicator]

        sns.boxplot(data=subset_indicator, x=x_column, y=y_column, ax=axes[l, c],
                    showfliers=False,
                    medianprops={"color": "coral", 'linewidth': 4.0},
                    showmeans=True,
                    meanprops={'marker': 'o', 'markeredgecolor': 'black',
                               'markerfacecolor': dico_param['markerfacecolor'], 'markersize': 20},
                    boxprops=dico_param['boxprops'],
                    capprops={'color': dico_param['color'], 'linewidth': 4.0},
                    whiskerprops={'color': dico_param['color'], 'linewidth': 4.0})

        axes[l, c].set_title(
            indicator, fontname='Arial Rounded MT Bold', color=rgb_text, fontsize=45, pad=50)
        axes[l, c].set_ylabel(y_column, fontsize=40,
                              fontname='Arial Rounded MT Bold', color=rgb_text)
        axes[l, c].set_xlabel(indicator + ' \nin ' + x_column, fontsize=40,
                              fontname='Arial Rounded MT Bold', color=rgb_text)
        axes[l, c].tick_params(axis='both', which='major',
                               labelsize=40, labelcolor=rgb_text)
        axes[l, c].xaxis.offsetText.set_fontsize(40)

        if with_scatter:
            vals, ys = [], []
            for i, region in enumerate(subset_indicator[y_column].unique()):
                subset_without_outliers = filter_outlier(
                    subset_indicator.loc[subset_indicator[y_column] == region], x_column)
                vals.append(subset_without_outliers[x_column])
                ys.append(np.random.normal(i, 0.05, len(
                    subset_without_outliers['Country Name'].unique())))

            for y, val in zip(ys, vals):
                axes[l, c].scatter(val, y, alpha=0.4, s=400)

        (c, l) = (0, l+1) if c == 1 else (c+1, l)

    plt.show()


def plot_boxplot_by(dataset: pd.DataFrame, by_column: str, year_list: list[str], limit: int, with_scatter: bool):
    """
    Construit pour chaque indicateur, un boxplot par groupe de pays 

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant les valeurs à utiliser dans les boxplots
    by_column : str : nom de la colonne selon laquelle regrouper les pays
    year_list : list of strings : liste des années à prendre en compte
    limit : int : nombre minimum de pays nécessaire par groupe pour contruire un boxplot
    with_scatters : bool : laisser ou non les valeurs individuelles apparentes
    """
    subset = reshape_dataset_forboxplot(dataset, by_column)

    if len(year_list) == 1:
        year = year_list[0]
    else:
        subset['mean (2000 to 2015)'] = subset[year_list].mean(axis=1)
        year = 'mean (2000 to 2015)'

    mask_not_enough_countries = subset[by_column + ' n'] < limit
    subset_not_enough_countries = subset.loc[mask_not_enough_countries]
    subset = subset.loc[(~mask_not_enough_countries)]

    if not subset_not_enough_countries.empty:
        print("\nNot enough countries in the following categories : ",
              subset_not_enough_countries[by_column].unique(), 'not relevant to plot a boxplot\n')

        subset_not_enough_countries = subset_not_enough_countries[[
            by_column + ' (n)', 'Country Name', year, 'Indicator Name']]
        subset_not_enough_countries = subset_not_enough_countries.pivot_table(index=[by_column + ' (n)', 'Country Name'],
                                                                              columns='Indicator Name', values=year)

        display(subset_not_enough_countries.head(
            subset_not_enough_countries.shape[0]))

    plot_boxplot_by_indicator(subset, year, by_column + ' (n)', with_scatter)


def get_columns_contains(dataset: pd.DataFrame, regex_str: str):
    """
    Retourne la liste des colonnes contenant une expression régulière donnée dans leur nom

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données à analyser
    regex_str : str : expression régulière recherchée
    """
    mask = dataset.columns.str.contains(regex_str, regex=True)
    columns_list = dataset.columns[mask].values

    return columns_list


def get_percentile_score(dataset: pd.DataFrame, value_column: str, new_score_column: str):
    """
    Retourne un dataframe après ajout d'une colonne contenant les rangs percentiles 

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données à modifier
    value_column : str : nom de la colonne sur laquelle calculer un rang percentile
    new_score_column : str : nom de la nouvelle colonne à ajouter (qui contient les rangs percentiles)
    """
    subset = dataset.sort_values(value_column)
    all_values = subset[value_column].values
    subset[new_score_column] = scipy.stats.percentileofscore(
        a=all_values, score=all_values, kind='rank')
    subset[new_score_column] = round(subset[new_score_column])

    return subset


def get_score(dataset: pd.DataFrame, weight_dict: {str: float}, percentile_score: bool):
    """
    Retourne un dataframe contenant pour chaque pays, un score pour chaque indicateur et le score d'attractivité moyen

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données initial contenant les pays et les valeurs par indicateur
    weight_dict : dict : poids à utiliser pour calculer le score moyen
    percentile_score : bool : calculer ou non les scores en utilisant la méthode du rang percentile
    """
    score = pd.DataFrame(index=dataset['Country Name'].unique())

    for year in get_columns_year(dataset):
        subset_year = dataset[['Country Name', 'Indicator Name', year]].copy()
        subset_year = subset_year.pivot_table(
            index='Country Name', columns='Indicator Name', values=year)

        for indicator, weight in weight_dict.items():
            if percentile_score:
                subset_year = get_percentile_score(
                    subset_year, indicator, 'SCORE_' + indicator + '_' + year)
            else:
                subset_year['SCORE_' + indicator + '_' + year] = (
                    subset_year[indicator] / subset_year[indicator].max()) * 100

            subset_year['WEIGHTEDSCORE_' + indicator + '_' + year] = weight * \
                subset_year['SCORE_' + indicator + '_' + year]

        subset_year['SCORE_SYNTH_' + year] = subset_year[get_columns_contains(
            subset_year, '^WEIGHTEDSCORE_')].sum(axis=1)
        score = score.merge(subset_year[get_columns_contains(
            subset_year, '^SCORE_')], right_index=True, left_index=True, how='left')

    score['SCORE_SYNTH_MEAN'] = score[get_columns_contains(
        score, '^SCORE_SYNTH_[0-9]{4}')].mean(axis=1)

    return score


def reshape_score_dataset_tolineplot(dataset: pd.DataFrame, year: str, rank_top: int):
    """
    Retourne un dataframe transformé pour faciliter la création d'un diagramme en lignes 
    à partir des scores d'attractivité des pays.

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données à transformer
    year : str : année de référence (du top n)
    rank_top : int : nombre de pays à afficher dans la légende comme étant le top n
    """
    columns_score = get_columns_contains(dataset, '^SCORE_SYNTH_[0-9]{4}')

    subset_score = dataset[columns_score]
    subset_score = subset_score.sort_values(
        'SCORE_SYNTH_' + year, ascending=False)
    countries_top = subset_score.iloc[:rank_top].index.values

    subset_score = subset_score.filter(items=countries_top, axis=0)
    subset_score = subset_score.rename(
        columns={k: k.replace('SCORE_SYNTH_', "") for k in columns_score})

    subset_score = subset_score.transpose()
    subset_score.reset_index(inplace=True)
    subset_score.rename(columns={"index": "Year"}, inplace=True)
    melted_subset = subset_score.melt(
        'Year', var_name='Top ' + str(rank_top) + ' (in ' + year + ')', value_name='scores')

    return melted_subset