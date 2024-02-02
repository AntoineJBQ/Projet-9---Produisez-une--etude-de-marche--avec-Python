import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def etude_fichier_complet(df):
    # Étape 1 : Aperçu des données
    print("Aperçu des premières lignes :")
    print(df.sample(3))
    
    # Étape 2 : Informations sur les colonnes et les types de données
    print("\nInformations sur les colonnes et les types de données :")
    print(df.info())
    
    # Étape 3 : Statistiques descriptives
    print("\nStatistiques descriptives :")
    print(df.describe())
    
    # Étape 4 : Gestion des données manquantes
    print("\nGestion des données manquantes :")
    print(df.isnull().sum())
    
    # Étape 5 : Analyse exploratoire des données (EDA)
    print("\nAnalyse exploratoire des données :")
    
    # Distribution des variables numériques
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], bins=20, kde=True)
        plt.title(f'Distribution de {col}')
        plt.show()
        
    # Analyse des variables catégoriques
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col)
        plt.title(f'Fréquence de {col}')
        plt.xticks(rotation=45)
        plt.show()


def etude_fichier(df):
    print("Nombre de colonnes :", df.shape)
    print()
    print("Le type est : \n", df.dtypes)
    print()
    print('Nombre de valeurs uniques :')
    print(df.nunique())
    print()
    print('Le nombre de valeurs manquantes :\n', df.isnull().sum())

def analyse_statistique(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df.describe(include='all')

    stats.loc['median'] = df[numeric_cols].median()
    stats.loc['skewness'] = df[numeric_cols].skew()
    stats.loc['kurtosis'] = df[numeric_cols].kurtosis()

    return stats

def identifier_differences_entre_cles(df1, df2, keys_df1=None, keys_df2=None):
    if keys_df1 is None:
        keys_df1 = [df1.columns[0]]
    if keys_df2 is None:
        keys_df2 = [df2.columns[0]]

    unique_keys_df1 = set(df1[keys_df1].drop_duplicates().apply(tuple, axis=1))
    unique_keys_df2 = set(df2[keys_df2].drop_duplicates().apply(tuple, axis=1))

    keys_only_in_df1 = unique_keys_df1 - unique_keys_df2
    keys_only_in_df2 = unique_keys_df2 - unique_keys_df1

    print("Nombre de clés absentes dans DF1:", len(keys_only_in_df1))
    print("Nombre de clés absentes dans DF2:", len(keys_only_in_df2))

def etudier_jointures(df1, df2):
    id_columns_df1 = [col for col in df1.columns if 'id' in col.lower()]
    id_columns_df2 = [col for col in df2.columns if 'id' in col.lower()]

    results = {}

    for on_column_df1 in id_columns_df1:
        for on_column_df2 in id_columns_df2:
            inner_unique_keys = len(pd.merge(df1, df2, how='inner', left_on=on_column_df1, right_on=on_column_df2).drop_duplicates(subset=on_column_df1))
            left_unique_keys = len(pd.merge(df1, df2, how='left', left_on=on_column_df1, right_on=on_column_df2).drop_duplicates(subset=on_column_df1))
            right_unique_keys = len(pd.merge(df1, df2, how='right', left_on=on_column_df1, right_on=on_column_df2).drop_duplicates(subset=on_column_df1))
            outer_unique_keys = len(pd.merge(df1, df2, how='outer', left_on=on_column_df1, right_on=on_column_df2).drop_duplicates(subset=on_column_df1))

            results[f"{on_column_df1} - {on_column_df2}"] = {
                'inner': inner_unique_keys,
                'left': left_unique_keys,
                'right': right_unique_keys,
                'outer': outer_unique_keys
            }

    for key, values in results.items():
        print(f"Colonnes de jointure : {key}")
        print("Nombre de clés uniques après jointure interne:", values['inner'])
        print("Nombre de clés uniques après jointure à gauche:", values['left'])
        print("Nombre de clés uniques après jointure à droite:", values['right'])
        print("Nombre de clés uniques après jointure externe:", values['outer'])
        print()

def traiter_valeurs_manquantes(df, method='mean', columns=None):
    if method == 'mean':
        if columns is None:
            return df.fillna(df.mean())
        else:
            return df.fillna(df.mean()[columns])
    elif method == 'median':
        if columns is None:
            return df.fillna(df.median())
        else:
            return df.fillna(df.median()[columns])
    elif method == 'mode':
        if columns is None:
            return df.fillna(df.mode().iloc[0])
        else:
            return df.fillna(df.mode().iloc[0][columns])
    else:
        return df.fillna(method=method)

def etude_outliers(df, seuil=2.0):
    ''' 
    Fonction pour détecter et afficher le pourcentage d'outliers dans toutes les colonnes numériques d'un DataFrame,
    en utilisant le Z-score.

    Paramètres :
    df (DataFrame) : DataFrame Pandas.
    seuil (float) : Seuil pour le Z-score utilisé pour définir un outlier. Par défaut à 2.0.
    '''

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        series = df[col]
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers_mask = z_scores > seuil
        outliers_percentage = (outliers_mask.mean() * 100).round(2)

        print(f"Le pourcentage de valeurs considérées comme des outliers en utilisant le Z-score au seuil {seuil} dans la colonne '{col}' est {outliers_percentage}%")

def plot_skewness_kurtosis(data, column):
    skewness = data[column].skew()
    kurtosis = data[column].kurtosis()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(data[column], kde=True)
    plt.axvline(x=data[column].mean(), color='r', linestyle='dashed', linewidth=2)
    plt.title(f'Skewness: {skewness:.2f}')

    plt.subplot(1, 2, 2)
    sns.histplot(data[column], kde=True)
    plt.axvline(x=data[column].mean(), color='r', linestyle='dashed', linewidth=2)
    plt.axvline(x=data[column].mean() + data[column].std(), color='g', linestyle='dashed', linewidth=2)
    plt.axvline(x=data[column].mean() - data[column].std(), color='g', linestyle='dashed', linewidth=2)
    plt.axvline(x=data[column].mean() + 2 * data[column].std(), color='b', linestyle='dashed', linewidth=2)
    plt.axvline(x=data[column].mean() - 2 * data[column].std(), color='b', linestyle='dashed', linewidth=2)
    plt.title(f'Kurtosis: {kurtosis:.2f}')

    plt.tight_layout()

    if skewness > 0:
        skewness_analysis = "La distribution est inclinée positivement vers la droite (queue à droite)."
    elif skewness < 0:
        skewness_analysis = "La distribution est inclinée positivement vers la gauche (queue à gauche)."
    else:
        skewness_analysis = "La distribution est parfaitement symétrique."

    if kurtosis > 0:
        kurtosis_analysis = "La distribution est leptokurtique, avec des pics plus fins et des queues plus épaisses."
    elif kurtosis < 0:
        kurtosis_analysis = "La distribution est platykurtique, avec des pics plus larges et des queues plus minces."
    else:
        kurtosis_analysis = "La distribution est mésokurtique, similaire à une distribution normale."

    print("Analyse de la répartition (skewness):", skewness_analysis)
    print("Analyse de l'aplatissement (kurtosis):", kurtosis_analysis)

    plt.show()

def correlation_matrix(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matrice de Corrélation")
    plt.show()

def visualiser_valeurs_manquantes(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Visualisation des Valeurs Manquantes")
    plt.show()

def encoder_variables_categorielles(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include='object').columns
    return pd.get_dummies(df, columns=columns, drop_first=True)

def reduire_dimension_pca(df, n_components=None):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df.select_dtypes(include=[np.number]))
    columns = [f"PC{i+1}" for i in range(components.shape[1])]
    df_pca = pd.DataFrame(components, columns=columns)
    return pd.concat([df.drop(columns=df.select_dtypes(include=[np.number]).columns), df_pca], axis=1)

def plot_skewness_kurtosis(data, column):
    skewness = data[column].skew()
    kurtosis = data[column].kurtosis()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(data[column], kde=True)
    plt.axvline(x=data[column].mean(), color='r', linestyle='dashed', linewidth=2)
    plt.title(f'Skewness: {skewness:.2f}')

    plt.subplot(1, 2, 2)
    sns.histplot(data[column], kde=True)
    plt.axvline(x=data[column].mean(), color='r', linestyle='dashed', linewidth=2)
    plt.axvline(x=data[column].mean() + data[column].std(), color='g', linestyle='dashed', linewidth=2)
    plt.axvline(x=data[column].mean() - data[column].std(), color='g', linestyle='dashed', linewidth=2)
    plt.axvline(x=data[column].mean() + 2 * data[column].std(), color='b', linestyle='dashed', linewidth=2)
    plt.axvline(x=data[column].mean() - 2 * data[column].std(), color='b', linestyle='dashed', linewidth=2)
    plt.title(f'Kurtosis: {kurtosis:.2f}')

    plt.tight_layout()

    if skewness > 0:
        skewness_analysis = "La distribution est inclinée positivement vers la droite (queue à droite)."
    elif skewness < 0:
        skewness_analysis = "La distribution est inclinée positivement vers la gauche (queue à gauche)."
    else:
        skewness_analysis = "La distribution est parfaitement symétrique."

    if kurtosis > 0:
        kurtosis_analysis = "La distribution est leptokurtique, avec des pics plus fins et des queues plus épaisses."
    elif kurtosis < 0:
        kurtosis_analysis = "La distribution est platykurtique, avec des pics plus larges et des queues plus minces."
    else:
        kurtosis_analysis = "La distribution est mésokurtique, similaire à une distribution normale."

    print("Analyse de la répartition (skewness):", skewness_analysis)
    print("Analyse de l'aplatissement (kurtosis):", kurtosis_analysis)

    plt.show()

def correlation_matrix(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matrice de Corrélation")
    plt.show()

def visualiser_valeurs_manquantes(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Visualisation des Valeurs Manquantes")
    plt.show()

def encoder_variables_categorielles(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include='object').columns
    return pd.get_dummies(df, columns=columns, drop_first=True)

def reduire_dimension_pca(df, n_components=None):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df.select_dtypes(include=[np.number]))
    columns = [f"PC{i+1}" for i in range(components.shape[1])]
    df_pca = pd.DataFrame(components, columns=columns)
    return pd.concat([df.drop(columns=df.select_dtypes(include=[np.number]).columns), df_pca], axis=1)

def plot_pca_correlation_circle(pca, features, x=0, y=1):
    fig, ax = plt.subplots(figsize=(10, 9))
    for i in range(0, pca.components_.shape[1]):
        ax.arrow(0,
                 0,  # Start the arrow at the origin
                 pca.components_[x, i],  # x for PCx
                 pca.components_[y, i],  # y for PCy
                 head_width=0.07,
                 head_length=0.07,
                 width=0.02)

        plt.text(pca.components_[x, i] + 0.05,
                 pca.components_[y, i] + 0.05,
                 features[i])

    # affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

    plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))

    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
    plt.axis('equal')
    plt.show(block=False)
    
from math import pi

def get_cluster_data(data, cluster_column, cluster_value):
    """Renvoie un sous-ensemble de données pour le cluster spécifié."""
    return data[data[cluster_column] == cluster_value]

def radar_plot(*cluster_data_list):
    """Crée un radar plot pour les variables numériques de plusieurs clusters."""
    for cluster_data in cluster_data_list:
        numeric_columns = cluster_data.select_dtypes(include='number').columns
        stats = cluster_data[numeric_columns].mean().tolist()
        stats += stats[:1]  # repeat the first value to close the circular graph
        angles = [n / float(len(numeric_columns)) * 2 * pi for n in range(len(numeric_columns))]
        angles += angles[:1]
        plt.polar(angles, stats)
        plt.fill(angles, stats, alpha=0.1)
    plt.xticks(angles[:-1], numeric_columns)
    plt.show()
    
def plot_boxplot(data):
    """Crée un boxplot pour toutes les colonnes numériques côte à côte."""
    numeric_columns = data.select_dtypes(include='number').columns
    sns.boxplot(data=data[numeric_columns])
    plt.xticks(rotation=90)  # Rotation des étiquettes sur l'axe des x pour une meilleure lisibilité
    plt.show()
    
def radar_plot_subplot(cluster_data_list, ax):
    """Crée un radar plot pour les variables numériques de plusieurs clusters."""
    for cluster_data in cluster_data_list:
        numeric_columns = cluster_data.select_dtypes(include='number').columns
        stats = cluster_data[numeric_columns].mean().tolist()
        stats += stats[:1]  # repeat the first value to close the circular graph
        angles = [n / float(len(numeric_columns)) * 2 * pi for n in range(len(numeric_columns))]
        angles += angles[:1]
        ax.plot(angles, stats)  # Utilisez plot au lieu de polar
        ax.fill(angles, stats, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(numeric_columns)

def plot_boxplot_subplot(data, ax):
    """Crée un boxplot pour toutes les colonnes numériques côte à côte."""
    numeric_columns = data.select_dtypes(include='number').columns
    sns.boxplot(data=data[numeric_columns], ax=ax)
    ax.set_xticklabels(numeric_columns, rotation=90)  # Rotation des étiquettes sur l'axe des x pour une meilleure lisibilité

def plot_subplots(cluster_data_list, data):
    """Crée un subplot avec un radar plot et un boxplot."""
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, polar=True)
    ax2 = fig.add_subplot(122)
    radar_plot_subplot(cluster_data_list, ax1)
    plot_boxplot_subplot(data, ax2)
    plt.show()
    
def descriptive_statistics(cluster_data):
    """Affiche les statistiques descriptives pour les colonnes numériques du cluster."""
    numeric_columns = cluster_data.select_dtypes(include='number')
    print(numeric_columns.describe())
    
def analyze_cluster(data, cluster_column, cluster_value):
    """Analyse un cluster spécifié en appelant toutes les fonctions."""
    cluster_data = get_cluster_data(data, cluster_column, cluster_value)
    print('Boxplot et radarplot pour le cluster', cluster_value)
    plot_subplots([cluster_data], data)
    # print("Boxplot des colonnes numériques :")
    # plot_boxplot(cluster_data)
    # print("Radar plot des colonnes numériques :")
    # radar_plot(cluster_data)
    # print("Statistiques descriptives des colonnes numériques :")
    # descriptive_statistics(cluster_data)
    
def plot_all_pca_correlation_circles(pca, features):
    n_components = pca.n_components_
    fig, axs = plt.subplots(n_components-1, n_components-1, figsize=(15, 15))

    for i in range(n_components):
        for j in range(i+1, n_components):
            ax = axs[i, j-1]  # j-1 car il n'y a pas de subplot pour i=j
            for k in range(0, pca.components_.shape[1]):
                ax.arrow(0, 0, pca.components_[i, k], pca.components_[j, k], head_width=0.07, head_length=0.07, width=0.02)
                ax.text(pca.components_[i, k] + 0.05, pca.components_[j, k] + 0.05, features[k])
            ax.plot([-1, 1], [0, 0], color='grey', ls='--')
            ax.plot([0, 0], [-1, 1], color='grey', ls='--')
            ax.set_xlabel('F{} ({}%)'.format(i+1, round(100*pca.explained_variance_ratio_[i],1)))
            ax.set_ylabel('F{} ({}%)'.format(j+1, round(100*pca.explained_variance_ratio_[j],1)))
            ax.set_title("Cercle des corrélations (F{} et F{})".format(i+1, j+1))
            an = np.linspace(0, 2 * np.pi, 100)
            ax.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
            ax.axis('equal')

    plt.tight_layout()
    plt.show(block=False)        