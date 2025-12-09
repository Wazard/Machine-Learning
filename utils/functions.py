import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

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


def get_target_correlation_matrix(df: pd.DataFrame, target: str, threshold: float = 0.001):
    """
    Display a correlation matrix between the target column and all other numeric columns,
    filtering out correlations below the threshold.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
    
    # Compute correlations
    corr_matrix = df.corr(numeric_only=True)
    
    # Extract correlations with target
    target_corr = corr_matrix[[target]].drop(index=target)
    
    # Filter by threshold
    filtered = target_corr[target_corr[target].abs() >= threshold]
    
    if filtered.empty:
        print(f"No correlations >= {threshold} found for target '{target}'.")
        return
    
    # Display as a matrix (heatmap)
    plt.figure(figsize=(6, len(filtered) * 0.4 + 2))
    sns.heatmap(filtered, annot=True, cmap="coolwarm", center=0, cbar=True)
    plt.title(f"Correlation Matrix with Target: {target}")
    plt.show()
    
    return filtered

def find_optimal_threshold(y_true, y_proba, metric = None):
    """
    Sweeps through potential decision thresholds (0.01 to 0.99) to find the one that
    maximizes the given classification metric (defaulting to F1-Score).
    
    This is essential for imbalanced datasets like the 62/38 split, as it selects
    a threshold that balances Precision and Recall for the positive class.
    
    Args:
        y_true (np.array): The true binary labels (0 or 1).
        y_proba (np.array): The predicted probabilities (0.0 to 1.0).
        metric (callable): The scoring function to maximize (e.g., f1_score, accuracy_score).
        
    Returns:
        float: The optimal threshold (the probability value that maximizes the metric).
        float: The maximum value achieved by that metric.
    """
    from sklearn.metrics import f1_score
    if metric is None:
        metric = f1_score

    # Test thresholds from 0.01 up to 0.99, in increments of 0.01
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_metric_value = 0.0
    best_threshold = 0.5
    
    for threshold in thresholds:
        # Convert probabilities to hard labels based on the current threshold
        y_pred_labels = np.where(y_proba >= threshold, 1, 0)
        
        # Calculate the metric value for this threshold
        # handle potential warning if prediction is all one class
        try:
            current_metric_value = metric(y_true, y_pred_labels)
        except ValueError:
            continue
            
        if current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            best_threshold = threshold
            
    # You can return both the threshold and the metric value for reporting
    return best_threshold, best_metric_value