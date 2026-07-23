import matplotlib.pyplot as plt
from cycler import cycler

#%% Defining color palettes and default sizes

pal_pink = [ '590d22', 'c9184a', 'ff4d6d', 'ff8fa3', 'ffb3c1', 'ffccd5', 'ffe0e6']
pal_pink = [f"#{c}" for c in pal_pink]

pal_blue = ['013a63',  '2a6f97', '468faf', '61a5c2', '89c2d9', 'a9d6e5']
pal_blue = [f"#{c}" for c in pal_blue]

DEFAULT_ANNOTATION_SIZE = 14
DEFAULT_TITLE_SIZE = 16

#%% Setting the style for the plots

def set_transparent_style(edge_color = 'black', palette=None, label_size=DEFAULT_ANNOTATION_SIZE, tick_label_size=DEFAULT_ANNOTATION_SIZE):
    """
    Set matplotlib global style with transparent background, white text/lines, and custom color palette.
    Parameters:
        palette (list of color strings, optional): List of colors to use as default line colors.
        label_size (int): Font size for axis labels (titles).
        tick_label_size (int): Font size for tick labels.
    """
    if palette is None:
        # Default palette: white lines only
        palette = [edge_color]

    plt.rcParams.update({
        'figure.facecolor': 'none',         # Transparent figure background
        'axes.facecolor': 'none',           # Transparent axes background
        'axes.edgecolor': edge_color,       # axes border
        'axes.labelcolor': edge_color,         # White axes labels
        'axes.labelsize': label_size,       # Axis label size (titles)
        'xtick.color': edge_color,             # White x tick labels
        'ytick.color': edge_color,             # White y tick labels
        'axes.titlecolor': edge_color,  # for title color
        'axes.titlesize': label_size,        # for title font size (example)
        'xtick.labelsize': tick_label_size, # X tick label font size
        'ytick.labelsize': tick_label_size, # Y tick label font size
        'grid.color': edge_color,              # White grid lines
        'text.color': edge_color,              # White text
        'lines.color': edge_color,             # Default line color white (fallback)
        'savefig.transparent': True,        # Savefig transparent background
        'axes.prop_cycle': cycler('color', palette)  # Set default line colors cycle
    })

# Function cleaning the spines of the plots for better aesthetics
def clean_plot(ax):
    """Clean plot spines for better aesthetics
        Parameters: ax : matplotlib.axes.Axes """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(left=False)
