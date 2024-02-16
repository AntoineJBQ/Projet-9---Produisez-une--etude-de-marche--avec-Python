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

import pandas as pd
from sklearn.decomposition import PCA

def apply_pca(X):
    """
    Applique l'Analyse en Composantes Principales (ACP) sur les données X.

    Parameters:
        X (DataFrame): Les données d'entrée.

    Returns:
        pca (PCA): L'objet PCA ajusté.
    """
    # Créer les composantes principales
    pca = PCA()
    X_acp = pca.fit_transform(X)
    # Convertir en dataframe
    noms_composantes = [f"CP{i+1}" for i in range(X_acp.shape[1])]
    X_acp = pd.DataFrame(X_acp, columns=noms_composantes)
    # Créer les chargements
    chargements = pd.DataFrame(
        pca.components_.T,  # transposer la matrice des chargements
        columns=noms_composantes,  # les colonnes sont les composantes principales
        index=X.columns,  # les lignes sont les variables originales
    )
    return pca

def plot_variance(acp, largeur=8, dpi=100):
    """
    Trace les graphiques de la variance expliquée et cumulative de l'ACP.

    Parameters:
        acp (PCA): L'objet PCA ajusté.
        largeur (int): La largeur de la figure.
        dpi (int): La résolution de la figure.

    Returns:
        axs (array): Les axes des graphiques.
    """
    # Créer la figure
    fig, axs = plt.subplots(1, 2)
    n = acp.n_components_
    grille = np.arange(1, n + 1)
    # Variance expliquée
    variance_exp = acp.explained_variance_ratio_
    axs[0].bar(grille, variance_exp)
    axs[0].set(
        xlabel="Composante", title="% Variance Expliquée", ylim=(0.0, 1.0)
    )
    # Variance cumulative
    variance_cumul = np.cumsum(variance_exp)
    axs[1].plot(np.r_[0, grille], np.r_[0, variance_cumul], "o-")
    axs[1].set(
        xlabel="Composante", title="% Variance Cumulative", ylim=(0.0, 1.0)
    )
    # Configurer la figure
    fig.set(figwidth=largeur, dpi=dpi)
    return axs

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

# Feature Scaling
def scale_features(df, scaler):
    """
    Scale numerical features using a specified scaler.

    Parameters:
        df (DataFrame): Input DataFrame with numerical features.
        scaler: Scaler object (e.g., StandardScaler, MinMaxScaler).

    Returns:
        DataFrame: DataFrame with scaled numerical features.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# Feature Engineering
def create_new_feature(df, feature_name, expression):
    """
    Create a new feature in the DataFrame based on a mathematical expression.

    Parameters:
        df (DataFrame): Input DataFrame.
        feature_name (str): Name of the new feature.
        expression (str): Mathematical expression for the new feature.

    Returns:
        DataFrame: DataFrame with the new feature added.
    """
    df[feature_name] = df.eval(expression)
    return df

# Model Evaluation
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of a classification model.

    Parameters:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        dict: Classification report and confusion matrix.
    """
    report = classification_report(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    return {'classification_report': report, 'confusion_matrix': matrix}

# Data Splitting
from sklearn.model_selection import train_test_split

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
        df (DataFrame): Input DataFrame.
        target_column (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generation.

    Returns:
        dict: Dictionary containing training and testing DataFrames.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target_column, axis=1), df[target_column],
        test_size=test_size, random_state=random_state
    )
    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

def tune_hyperparameters(model, param_grid, X, y, cv=5):
    """
    Tune hyperparameters using GridSearchCV.

    Parameters:
        model: Machine learning model.
        param_grid (dict): Dictionary with hyperparameter values.
        X: Features.
        y: Target variable.
        cv (int): Number of cross-validation folds.

    Returns:
        GridSearchCV: Fitted GridSearchCV object.
    """
    grid_search = GridSearchCV(model, param_grid, cv=cv)
    grid_search.fit(X, y)
    return grid_search

# Time Series Analysis (Example: Rolling Mean)
def time_series_rolling_mean(series, window_size):
    """
    Calculate the rolling mean for a time series.

    Parameters:
        series (Series): Time series data.
        window_size (int): Size of the rolling window.

    Returns:
        Series: Time series with rolling mean.
    """
    return series.rolling(window=window_size).mean()

# Text Data Processing
from sklearn.feature_extraction.text import CountVectorizer

def tokenize_text_data(text_data):
    """
    Tokenize text data using CountVectorizer.

    Parameters:
        text_data (Series): Series containing text data.

    Returns:
        DataFrame: Tokenized representation of text data.
    """
    vectorizer = CountVectorizer()
    tokenized_data = vectorizer.fit_transform(text_data)
    return pd.DataFrame(tokenized_data.toarray(), columns=vectorizer.get_feature_names_out())

# Ensemble Methods
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

def bagging_classifier(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a BaggingClassifier.

    Parameters:
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.

    Returns:
        BaggingClassifier: Fitted BaggingClassifier.
    """
    bagging_classifier = BaggingClassifier()
    bagging_classifier.fit(X_train, y_train)
    accuracy = bagging_classifier.score(X_test, y_test)
    print(f'Bagging Classifier Accuracy: {accuracy:.2f}')
    return bagging_classifier

def adaboost_classifier(X_train, y_train, X_test, y_test):
    """
    Train and evaluate an AdaBoostClassifier.

    Parameters:
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.

    Returns:
        AdaBoostClassifier: Fitted AdaBoostClassifier.
    """
    adaboost_classifier = AdaBoostClassifier()
    adaboost_classifier.fit(X_train, y_train)
    accuracy = adaboost_classifier.score(X_test, y_test)
    print(f'AdaBoost Classifier Accuracy: {accuracy:.2f}')
    return adaboost_classifier

# Model Deployment (Example: Pickle)
import pickle

def save_model(model, filename):
    """
    Save a trained model using Pickle.

    Parameters:
        model: Trained machine learning model.
        filename (str): Name of the file to save the model.

    Returns:
        None
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    """
    Load a trained model using Pickle.

    Parameters:
        filename (str): Name of the file containing the saved model.

    Returns:
        Model: Loaded machine learning model.
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model


# Custom Visualization (Example: Correlation Heatmap)
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df):
    """
    Plot a correlation heatmap for the DataFrame.

    Parameters:
        df (DataFrame): Input DataFrame.

    Returns:
        None
    """
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()


import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose

def handle_missing_data(df, strategy='mean'):
    """
    Handle missing data in a DataFrame.

    Parameters:
        df (DataFrame): Input DataFrame.
        strategy (str): Strategy for imputing missing values ('mean', 'median', 'mode', 'constant').

    Returns:
        DataFrame: DataFrame with missing values handled.
    """
    if strategy == 'mean':
        df.fillna(df.mean(), inplace=True)
    elif strategy == 'median':
        df.fillna(df.median(), inplace=True)
    elif strategy == 'mode':
        df.fillna(df.mode().iloc[0], inplace=True)
    elif strategy == 'constant':
        df.fillna(0, inplace=True)  # Replace with desired constant value
    return df

def handle_outliers(df, z_threshold=3):
    """
    Identify and handle outliers in a DataFrame using Z-score.

    Parameters:
        df (DataFrame): Input DataFrame.
        z_threshold (float): Z-score threshold for identifying outliers.

    Returns:
        DataFrame: DataFrame with outliers handled.
    """
    z_scores = np.abs(zscore(df))
    df_no_outliers = df[(z_scores < z_threshold).all(axis=1)]
    return df_no_outliers

def plot_feature_importance(X, y):
    """
    Plot feature importance using a RandomForestRegressor.

    Parameters:
        X: Features.
        y: Target variable.

    Returns:
        None
    """
    model = RandomForestRegressor()
    model.fit(X, y)
    feature_importance = model.feature_importances_
    features = X.columns
    plt.bar(features, feature_importance)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.show()

def decompose_time_series(series, freq=12):
    """
    Decompose a time series into trend, seasonal, and residual components.

    Parameters:
        series (Series): Time series data.
        freq (int): Seasonal decomposition frequency.

    Returns:
        tuple: Trend, seasonal, and residual components.
    """
    decomposition = seasonal_decompose(series, freq=freq)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    return trend, seasonal, residual
