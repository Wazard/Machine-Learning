import matplotlib.pyplot as plt
import math
import numpy as np

def get_prediction_graph(models: dict, 
                        y_test, 
                        n_rows: int = 1, 
                        n_cols: int = None, 
                        base_cell_size: tuple[float, float] = (6, 4), 
                        zoom: float = 1.0, 
                        scatter_color: str = "royalblue", 
                        line_color: str = "red"):
    '''
    Returns fig and axes predictions vs values for regression algorithms.
    
    models MUST be a dictionary formatted as:
    models = {"model_name": (model:pipeline|None, model_y_pred_test)}

    Parameters
    ----------
    base_cell_size : tuple
        Width, height of each subplot cell in inches (default=(6,4)).
    zoom : float
        Multiplier to zoom in/out the figure size (default=1.0).
    scatter_color : str
        Color for scatter points (default="royalblue").
    line_color : str
        Color for the perfect prediction line (default="red").
    '''

    if n_cols is None:
        max_items = len(models)
        n_cols = min(4, max_items)               # up to 4 columns
        n_rows = math.ceil(max_items / n_cols)   # enough rows to fit everything

    # automatically compute figsize based on grid size
    figsize = (base_cell_size[0] * n_cols * zoom,
               base_cell_size[1] * n_rows * zoom)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # ensure axes is always iterable
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = [axes]

    for ax, (name, (model, y_pred)) in zip(axes, models.items()):
        ax.scatter(y_test, y_pred, alpha=0.4, color=scatter_color,
                   edgecolor='k', s=20)
        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                linestyle="--", color=line_color, lw=2,
                label="Perfect Prediction")
        ax.set_xlabel("Real Value")
        ax.set_ylabel("Predicted Value")
        ax.set_title(f"{name} Test Performance")
        ax.legend()
        ax.grid(True, alpha=0.3)

    return fig, axes