
# data visualization imports
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import seaborn as sns


def graficar_distribuciones(df, columna_grupo, columna_metrica):
    fig, axes = plt.subplots(df[columna_grupo].nunique(),1,figsize=(7,2.5*df[columna_grupo].nunique()))
    for ax, grupo in zip(axes.flat, df[columna_grupo].unique()):
        ax.set_title(f"Distribución de minutos de montaje para el método {grupo}")
        grupo_df = df[df[columna_grupo] == grupo]
        sns.histplot(grupo_df[columna_metrica], ax=ax, bins=25)

    plt.tight_layout()
    plt.show()



def plot_line_labels(ax: plt.Axes, interval: int = 1, contrast: bool = False) -> None:
    """
    Adds labels to each line in the plot at specified intervals.

    Parameters:
    ----------
    ax : plt.Axes
        Matplotlib Axes object where the line labels will be added.
    interval : int
        Interval for labeling data points on the line.
    contrast : bool
        Whether to adjust text color based on line color brightness for better readability.
    """
    if not isinstance(contrast, bool):
        raise TypeError(f"Expected 'contrast' to be of type 'bool', but got {type(contrast).__name__} instead.")
    
    for line in ax.lines:
        line_color = line.get_color()

        if contrast:
            r, g, b = line.get_color()[:3]
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            text_color = 'white' if brightness < 0.5 else 'black'
        else:
            text_color = 'white'

        for it, (x_data, y_data) in enumerate(zip(line.get_xdata(), line.get_ydata())):
            if it % interval == 0:
                ax.text(
                    x_data, y_data, f'{y_data:.0f}', 
                    ha='center', va='bottom',
                    color=text_color,
                    bbox=dict(facecolor=line_color, edgecolor='none', alpha=0.8)
                )


def plot_bar_labels(ax: plt.Axes, contrast: bool = False) -> None:
    """
    Adds labels to each bar in the plot.

    Parameters:
    ----------
    ax : plt.Axes
        Matplotlib Axes object where the bar labels will be added.
    contrast : bool
        Whether to adjust text color based on bar color brightness for better readability.
    """
    if not isinstance(contrast, bool):
        raise TypeError(f"Expected 'contrast' to be of type 'bool', but got {type(contrast).__name__} instead.")
    
    for bar in ax.patches:
        height = bar.get_height()
        
        if height > 0.01:
            bar_color = bar.get_facecolor()
            
            if contrast:
                r, g, b = bar_color[:3]
                brightness = (r * 299 + g * 587 + b * 114) / 1000
                text_color = 'white' if brightness < 0.5 else 'black'
            else:
                text_color = 'white'

            x_position = bar.get_x() + bar.get_width() / 2
            ax.text(
                x_position, height / 2, f'{height:.2f}',
                ha='center', va='center', color=text_color,
                bbox=dict(facecolor=bar_color, edgecolor='none', alpha=0.8),
                fontsize=8
            )


def create_time_xticks(ax: plt.Axes, hour_interval: int = 1, format: str = '%m/%d %H:00', rotation: int = 45) -> None:
    """
    Formats x-axis with time-based tick intervals.

    Parameters:
    ----------
    ax : plt.Axes
        Matplotlib Axes object to set time-based x-ticks.
    hour_interval : int
        Interval in hours for x-axis ticks.
    format : str
        Date format for the x-axis labels.
    rotation : int
        Rotation angle for x-axis labels.
    """
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=hour_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(format))
    ax.tick_params(axis='x', rotation=rotation)
