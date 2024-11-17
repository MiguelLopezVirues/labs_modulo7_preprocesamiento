import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import math


def separar_dataframe(dataframe):
    return (dataframe.select_dtypes(include=np.number),dataframe.select_dtypes(include=["O","category"]))

def plot_univariante_numerica(df, ncols):
    columnas_numericas = df.select_dtypes(include = np.number).columns
    nrows = math.ceil(len(columnas_numericas)/ncols)

    fig, axes = plt.subplots(nrows = nrows, ncols=ncols, figsize=(15,4*nrows))
    axes_flat = axes.flat

    for ax, columna in zip(axes_flat, columnas_numericas):
        if df[columna].nunique() < 30:
            plot = sns.countplot(data = df, x=columna, ax=ax)
            title = "Countplot"
        else:
            plot = sns.histplot(data = df, x=columna, ax=ax, bins=20)
            title = "Histplot"

        ax.set_title(title + f" de la variable {columna}")
    
    plt.suptitle("Distribución de variables numéricas")
    
    empty_subplots = ncols * nrows - len(columnas_numericas)
    if empty_subplots > 0:
        for idx in range(1,empty_subplots+1):
            fig.delaxes(axes_flat[-idx])

    plt.tight_layout()
    plt.show()


class Visualizador:
    """
    Clase para visualizar la distribución de variables numéricas y categóricas de un DataFrame.

    Attributes:
    - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.

    Methods:
    - __init__: Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.
    - separar_dataframes: Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.
    - plot_numericas: Grafica la distribución de las variables numéricas del DataFrame.
    - plot_categoricas: Grafica la distribución de las variables categóricas del DataFrame.
    - plot_relacion2: Visualiza la relación entre una variable y todas las demás, incluyendo variables numéricas y categóricas.
    """

    def __init__(self, dataframe):
        """
        Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.

        Parameters:
        - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        """
        self.dataframe = dataframe

    def separar_dataframes(self):
        """
        Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.

        Returns:
        - pandas.DataFrame: DataFrame con variables numéricas.
        - pandas.DataFrame: DataFrame con variables categóricas.
        """
        return self.dataframe.select_dtypes(include=np.number), self.dataframe.select_dtypes(include="O")
    
    def plot_numericas(self, color="grey", tamano_grafica=(15, 5)):
        """
        Grafica la distribución de las variables numéricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
        lista_num = self.separar_dataframes()[0].columns
        _, axes = plt.subplots(nrows = 2, ncols = math.ceil(len(lista_num)/2), figsize=tamano_grafica, sharey=True)
        axes = axes.flat
        for indice, columna in enumerate(lista_num):
            sns.histplot(x=columna, data=self.dataframe, ax=axes[indice], color=color, bins=20)
        plt.suptitle("Distribución de variables numéricas")
        plt.tight_layout()

    def plot_categoricas(self, color="grey", tamano_grafica=(40, 10)):
        """
        Grafica la distribución de las variables categóricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
        dataframe_cat = self.separar_dataframes()[1]
        _, axes = plt.subplots(math.ceil(len(dataframe_cat.columns) / 2), 2, figsize=tamano_grafica)
        axes = axes.flat
        for indice, columna in enumerate(dataframe_cat.columns):
            sns.countplot(x=columna, data=self.dataframe, order=self.dataframe[columna].value_counts().index,
                          ax=axes[indice], color=color)
            axes[indice].tick_params(rotation=90)
            axes[indice].set_title(columna)
            axes[indice].set(xlabel=None)

        plt.tight_layout(pad=0.005)

        plt.suptitle("Distribución de variables categóricas", y=1.02, fontsize=16)

    def plot_relacion(self, vr, tamano_grafica=(40, 12), color="grey"):
        """
        Visualiza la relación entre una variable y todas las demás, incluyendo variables numéricas y categóricas.

        Parameters:
            - vr (str): El nombre de la variable en el eje y.
            - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (40, 12).
            - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        Returns:
            No devuelve nada    
        """
        df_numericas = self.separar_dataframes()[0].columns
        meses_ordenados = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig, axes = plt.subplots(3, int(len(self.dataframe.columns) / 3), figsize=tamano_grafica)
        axes = axes.flat

        for ax, columna in zip(axes,self.dataframe.columns):
            if columna == vr:
                fig.delaxes(ax)
            elif columna in df_numericas:
                sns.scatterplot(x=vr, 
                                y=columna, 
                                data=self.dataframe, 
                                color=color, 
                                ax=ax)
                ax.set_title(columna)
                ax.set(xlabel=None)
            else:
                if columna == "Month":
                    sns.barplot(x=columna, y=vr, data=self.dataframe, order=meses_ordenados, ax=ax,
                                color=color)
                    ax.tick_params(rotation=90)
                    ax.set_title(columna)
                    ax.set(xlabel=None)
                else:
                    sns.barplot(x=columna, y=vr, data=self.dataframe, ax=ax, color=color)
                    ax.tick_params(rotation=90)
                    ax.set_title(columna)
                    ax.set(xlabel=None)

        plt.tight_layout()
    
    def analisis_temporal(self, var_respuesta, var_temporal, color = "black", order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]):


        """
        Realiza un análisis temporal mensual de una variable de respuesta en relación con una variable temporal. Visualiza un gráfico de líneas que muestra la relación entre la variable de respuesta y la variable temporal (mes), con la línea de la media de la variable de respuesta.


        Params:
        -----------
        dataframe : pandas DataFrame. El DataFrame que contiene los datos.
        var_respuesta : str. El nombre de la columna que contiene la variable de respuesta.
        var_temporal : str. El nombre de la columna que contiene la variable temporal (normalmente el mes).
        order : list, opcional.  El orden de los meses para representar gráficamente. Por defecto, se utiliza el orden estándar de los meses.

        Returns:
        --------
        None

 
        """


        plt.figure(figsize = (15, 5))

        # Convierte la columna "Month" en un tipo de datos categórico con el orden especificado
        self.dataframe[var_temporal] = pd.Categorical(self.dataframe[var_temporal], categories=order, ordered=True)

        # Trama el gráfico
        sns.lineplot(x=var_temporal, 
                     y=var_respuesta, 
                     data=self.dataframe, 
                     color = color)

        # Calcula la media de PageValues
        mean_page_values = self.dataframe[var_respuesta].mean()

        # Agrega la línea de la media
        plt.axhline(mean_page_values, 
                    color='green', 
                    linestyle='--', 
                    label='Media de PageValues')


        # quita los ejes de arriba y de la derecha
        sns.despine()

        # Rotula el eje x
        plt.xlabel("Month");


    def deteccion_outliers(self, color = "grey"):

        """
        Detecta y visualiza valores atípicos en un DataFrame.

        Params:
            - dataframe (pandas.DataFrame):  El DataFrame que se va a usar

        Returns:
            No devuelve nada

        Esta función selecciona las columnas numéricas del DataFrame dado y crea un diagrama de caja para cada una de ellas para visualizar los valores atípicos.
        """

        lista_num = self.separar_dataframes()[0].columns

        fig, axes = plt.subplots(2, ncols = math.ceil(len(lista_num)/2), figsize=(15,5))
        axes = axes.flat

        for indice, columna in enumerate(lista_num):
            sns.boxplot(x=columna, data=self.dataframe, 
                        ax=axes[indice], 
                        color=color, 
                        flierprops={'markersize': 4, 'markerfacecolor': 'orange'})

        if len(lista_num) % 2 != 0:
            fig.delaxes(axes[-1])

        
        plt.tight_layout()

    def correlacion(self, tamano_grafica = (7, 5)):

        """
        Visualiza la matriz de correlación de un DataFrame utilizando un mapa de calor.

        Params:
            - dataframe : pandas DataFrame. El DataFrame que contiene los datos para calcular la correlación.

        Returns:
        No devuelve nada

        Muestra un mapa de calor de la matriz de correlación.

        - Utiliza la función `heatmap` de Seaborn para visualizar la matriz de correlación.
        - La matriz de correlación se calcula solo para las variables numéricas del DataFrame.
        - La mitad inferior del mapa de calor está oculta para una mejor visualización.
        - Permite guardar la imagen del mapa de calor como un archivo .png si se solicita.

        """

        plt.figure(figsize = tamano_grafica )

        mask = np.triu(np.ones_like(self.dataframe.corr(numeric_only=True), dtype = np.bool_))

        sns.heatmap(data = self.dataframe.corr(numeric_only = True), 
                    annot = True, 
                    vmin=-1,
                    vmax=1,
                    cmap="viridis",
                    linecolor="black", 
                    fmt='.1g', 
                    mask = mask)
    

