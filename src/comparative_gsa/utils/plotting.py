import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autoemulate.core.plotting import display_figure

def plot_sa_heatmap(
        results: pd.DataFrame,
        index: str = "ST",
        top_n: int | None = None,
        cmap: str = "coolwarm",
        normalize: bool = True,
        figsize: tuple | None = None,
        fname: str | None = None,
    ):
        """
        Plot a normalized Sobol sensitivity analysis heatmap.

        Parameters
        ----------
        results: pd.DataFrame
            Sensitivity index dataframe with columns ['index', 'parameter',
            'output', 'value'].
        index: str
            The type of sensitivity index to plot (e.g., 'ST').
        top_n: int | None
            Number of top parameters to include. If None, returns all. Defaults to
            None.
        cmap: str
            Matplotlib colormap. Defaults to 'coolwarm'.
        normalize: bool
            Wheterto normalize values to [0, 1]. Defaults to True.
        figsize: tuple | None
            Figure size as (width, height) in inches. Defaults to None.
        fname: str | None
            If provided, saves the figure to this file path. Defaults to None.
        """
        # Determine which parameters to include
        parameter_list = top_n_sobol_params(
            results,
            top_n=len(results["parameter"].unique()) if top_n is None else top_n,
        )

        fig = _plot_sa_heatmap(
            results, index, parameter_list, cmap, normalize, fig_size=figsize
        )

        if fname is None:
            return display_figure(fig)
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        return None

def _plot_sa_heatmap(
    si_df, index, parameters, cmap="coolwarm", normalize=True, fig_size=None
) -> matplotlib.figure.Figure:
    """
    Plot a sensitivity analysis heatmap for a given index.

    Parameters
    ----------
    results: pd.DataFrame
        Sensitivity index dataframe with columns ['index', 'parameter',
        'output', 'value'].
    index: str
        The type of sensitivity index to plot (e.g., 'ST').
    top_n: int | None
        Number of top parameters to include. If None, returns all. Defaults to
        None.
    cmap: str
        Matplotlib colormap. Defaults to 'coolwarm'.
    normalize: bool
        Wheterto normalize values to [0, 1]. Defaults to True.
    figsize: tuple | None
        Figure size as (width, height) in inches. Defaults to None.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure containing the SA heatmap.
    """
    # Filter the dataframe for the specified index
    df = si_df[si_df["index"] == index]

    # Pivot the dataframe to get a matrix: rows = outputs, cols = parameters
    heatmap_df = (
        df[df["parameter"].isin(parameters)]
        .pivot_table(
            index="output", columns="parameter", values="value", fill_value=np.nan
        )
        .reindex(columns=parameters)  # Ensure column order
    )

    # Normalize if requested
    if normalize:
        min_value = heatmap_df.min().min()
        max_value = heatmap_df.max().max()
        value_range = max_value - min_value if max_value != min_value else 1
        heatmap_df = (heatmap_df - min_value) / value_range

    # Convert to NumPy array
    data_np = heatmap_df.to_numpy()

    # layout - add space for legend
    nrows, ncols = _calculate_layout(data_np.shape[1], data_np.shape[0])
    fig_size = fig_size or (4.5 * ncols, 4.5 * nrows + 2)  # Extra width for legend

    # Plotting
    fig, ax = plt.subplots(figsize=fig_size)
    cax = ax.imshow(data_np, cmap=cmap, aspect="auto")

    # Colorbar
    cbar = fig.colorbar(cax, ax=ax)
    cbar_label = "Normalized Sensitivity" if normalize else "Sensitivity"
    cbar.set_label(cbar_label, rotation=270, labelpad=15)

    # Labels and ticks
    ax.set_title(f"{index} Sensitivity Analysis Heatmap", fontsize=14, pad=12)
    ax.set_xlabel("Parameters", fontsize=12)
    ax.set_ylabel("Outputs", fontsize=12)

    ax.set_xticks(np.arange(len(parameters)))
    ax.set_xticklabels(parameters, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index)

    # Gridlines
    ax.set_xticks(np.arange(-0.5, len(parameters), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(heatmap_df.index), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()

    return fig

def _calculate_layout(n_outputs: int, ncols: int | None = None):
    """Calculate plot layout (n rows, n cols)."""
    if ncols is None:
        ncols = 3 if n_outputs >= 3 else n_outputs
    nrows = int(np.ceil(n_outputs / ncols))
    return nrows, ncols

def top_n_sobol_params(
    sa_results_df: pd.DataFrame, top_n: int, sa_index: str = "ST"
) -> list[str]:
    """
    Return `top_n` most important parameters given Sobol sensitivity analysis.

    In case of multiple outputs, averages over them to rank the parameters.

    Parameters
    ----------
    sa_results_df: pd.DataFrame
        Dataframe results by `SensitivityAnalysis().run()`
    top_n: int
        Number of parameters to return.
    sa_index: str
        Which sensitivity index to rank the parameters by. One of ["S1", "S2", "ST].

    Returns
    -------
    list[str]
        List of `top_n` parameter names.
    """
    if not all(
        col in sa_results_df.columns for col in ["index", "parameter", "value"]
    ):
        msg = (
            "sa_results_df is missing required columns: 'index', 'parameter',"
            "or 'value'"
        )
        raise ValueError(msg)

    st_results = sa_results_df[sa_results_df["index"] == sa_index]

    # each parameter is evalued against each output
    # to rank parameters, average over how sensitive all outputs are to it
    top_n = (
        st_results.groupby("parameter")["value"]  # pyright: ignore[reportCallIssue, reportAssignmentType]
        .mean()
        .nlargest(top_n)
        .index.tolist()
    )
    assert isinstance(top_n, list)
    return top_n
