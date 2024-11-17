import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest # para hacer el ztest
import scikit_posthocs as sp


from src import data_visualization_support as dvs


class Asunciones:
    def __init__(self, dataframe, columna_numerica):

        self.dataframe = dataframe
        self.columna_numerica = columna_numerica
        
    

    def identificar_normalidad(self, metodo='shapiro', alpha=0.05, verbose=True):
        """
        Evalúa la normalidad de una columna de datos de un DataFrame utilizando la prueba de Shapiro-Wilk o Kolmogorov-Smirnov.

        Parámetros:
            metodo (str): El método a utilizar para la prueba de normalidad ('shapiro' o 'kolmogorov').
            alpha (float): Nivel de significancia para la prueba.
            verbose (bool): Si se establece en True, imprime el resultado de la prueba. Si es False, Returns el resultado.

        Returns:
            bool: True si los datos siguen una distribución normal, False de lo contrario.
        """

        if metodo == 'shapiro':
            _, p_value = stats.shapiro(self.dataframe[self.columna_numerica])
            resultado = p_value > alpha
            mensaje = f"los datos siguen una distribución normal según el test de Shapiro-Wilk." if resultado else f"los datos no siguen una distribución normal según el test de Shapiro-Wilk."
        
        elif metodo == 'kolmogorov':
            _, p_value = stats.kstest(self.dataframe[self.columna_numerica], 'norm')
            resultado = p_value > alpha
            mensaje = f"los datos siguen una distribución normal según el test de Kolmogorov-Smirnov." if resultado else f"los datos no siguen una distribución normal según el test de Kolmogorov-Smirnov."
        else:
            raise ValueError("Método no válido. Por favor, elige 'shapiro' o 'kolmogorov'.")

        if verbose:
            print(f"Para la columna {self.columna_numerica}, {mensaje}")
        else:
            return resultado

        
    def identificar_homogeneidad (self,  columna_categorica):
        
        """
        Evalúa la homogeneidad de las varianzas entre grupos para una métrica específica en un DataFrame dado.

        Parámetros:
        - columna (str): El nombre de la columna que se utilizará para dividir los datos en grupos.
        - columna_categorica (str): El nombre de la columna que se utilizará para evaluar la homogeneidad de las varianzas.

        Returns:
        No Returns nada directamente, pero imprime en la consola si las varianzas son homogéneas o no entre los grupos.
        Se utiliza la prueba de Levene para evaluar la homogeneidad de las varianzas. Si el valor p resultante es mayor que 0.05,
        se concluye que las varianzas son homogéneas; de lo contrario, se concluye que las varianzas no son homogéneas.
        """
        
        # lo primero que tenemos que hacer es crear tantos conjuntos de datos para cada una de las categorías que tenemos, Control Campaign y Test Campaign
        valores_evaluar = []
        
        for valor in self.dataframe[columna_categorica].unique():
            valores_evaluar.append(self.dataframe[self.dataframe[columna_categorica]== valor][self.columna_numerica])

        statistic, p_value = stats.levene(*valores_evaluar)
        if p_value > 0.05:
            print(f"En la variable {columna_categorica} las varianzas son homogéneas entre grupos.")
        else:
            print(f"En la variable {columna_categorica} las varianzas NO son homogéneas entre grupos.")




class TestEstadisticos:
    def __init__(self, dataframe, variable_respuesta, columna_categorica):
        """
        Inicializa la instancia de la clase TestEstadisticos.

        Parámetros:
        - dataframe: DataFrame de pandas que contiene los datos.
        - variable_respuesta: Nombre de la variable respuesta.
        - columna_categorica: Nombre de la columna que contiene las categorías para comparar.
        """
        self.dataframe = dataframe
        self.variable_respuesta = variable_respuesta
        self.columna_categorica = columna_categorica

    def generar_grupos(self):
        """
        Genera grupos de datos basados en la columna categórica.

        Retorna:
        Una lista de nombres de las categorías.
        """
        lista_categorias =[]
    
        for value in self.dataframe[self.columna_categorica].unique():
            variable_name = value  # Asigna el nombre de la variable
            variable_data = self.dataframe[self.dataframe[self.columna_categorica] == value][self.variable_respuesta].values.tolist()
            globals()[variable_name] = variable_data  
            lista_categorias.append(variable_name)
    
        return lista_categorias

    def comprobar_pvalue(self, pvalor):
        """
        Comprueba si el valor p es significativo.

        Parámetros:
        - pvalor: Valor p obtenido de la prueba estadística.
        """
        if pvalor < 0.05:
            print("Hay una diferencia significativa entre los grupos")
        else:
            print("No hay evidencia suficiente para concluir que hay una diferencia significativa.")

    def test_manwhitneyu(self, categorias): # SE PUEDE USAR SOLO PARA COMPARAR DOS GRUPOS, PERO NO ES NECESARIO QUE TENGAN LA MISMA CANTIDAD DE VALORES
        """
        Realiza el test de Mann-Whitney U.

        Parámetros:
        - categorias: Lista de nombres de las categorías a comparar.
        """
        statistic, p_value = stats.mannwhitneyu(*[globals()[var] for var in categorias])

        print("Estadístico del Test de Mann-Whitney U:", statistic)
        print("Valor p:", p_value)

        self.comprobar_pvalue(p_value)

    def test_wilcoxon(self, categorias): # SOLO LO PODEMOS USAR SI QUEREMOS COMPARAR DOS CATEGORIAS Y SI TIENEN LA MISMA CANTIDAD DE VALORES 
        """
        Realiza el test de Wilcoxon.

        Parámetros:
        - categorias: Lista de nombres de las categorías a comparar.
        """
        statistic, p_value = stats.wilcoxon(*[globals()[var] for var in categorias])

        print("Estadístico del Test de Wilcoxon:", statistic)
        print("Valor p:", p_value)

        # Imprime el estadístico y el valor p
        print("Estadístico de prueba:", statistic)
        print("Valor p:", p_value) 

        self.comprobar_pvalue(p_value)

    def test_kruskal(self, categorias):
       """
       Realiza el test de Kruskal-Wallis.

       Parámetros:
       - categorias: Lista de nombres de las categorías a comparar.
       """
       statistic, p_value = stats.kruskal(*[globals()[var] for var in categorias])

       print("Estadístico de prueba:", statistic)
       print("Valor p:", p_value)

       self.comprobar_pvalue(p_value)

    
    def test_anova(self, categorias):
        """
        Realiza el test ANOVA.

        Parámetros:
        - categorias: Lista de nombres de las categorías a comparar.
        """
        statistic, p_value = stats.f_oneway(*[globals()[var] for var in categorias])

        print("Estadístico F:", statistic)
        print("Valor p:", p_value)

        self.comprobar_pvalue(p_value)

    def post_hoc(self):
        """
        Realiza el test post hoc de Tukey.
        
        Retorna:
        Un DataFrame con las diferencias significativas entre los grupos.
        """
        resultado_posthoc =  pairwise_tukeyhsd(self.dataframe[self.variable_respuesta], self.dataframe[self.columna_categorica])
        tukey_df =  pd.DataFrame(data=resultado_posthoc._results_table.data[1:], columns=resultado_posthoc._results_table.data[0])
        tukey_df['group_diff'] = tukey_df['group1'] + '-' + tukey_df['group2']
        return tukey_df[tukey_df["p-adj"] <= 0.05][['meandiff', 'p-adj', 'lower', 'upper', 'group_diff']]

    def run_all_tests(self):
        """
        Ejecuta todos los tests estadísticos disponibles en la clase.

        Parámetros:
        - categorias: Lista de nombres de las categorías a comparar.
        """
        print("Generando grupos...")
        categorias_generadas = self.generar_grupos()
        print("Grupos generados:", categorias_generadas)


        test_methods = {
            "mannwhitneyu": self.test_manwhitneyu,
            "wilcoxon": self.test_wilcoxon,
            "kruskal": self.test_kruskal,
            "anova": self.test_anova
        }

        test_choice = input("¿Qué test desea realizar? (mannwhitneyu, wilcoxon, kruskal, anova): ").strip().lower()
        test_method = test_methods.get(test_choice)
        if test_method:
            print(f"\nRealizando test de {test_choice.capitalize()}...")
            test_method(categorias_generadas)
        else:
            print("Opción de test no válida.")
        
        print("Los resultados del test de Tukey son: \n")
        display(self.post_hoc())
