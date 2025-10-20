import sys
import pandas as pd
import ft_mlp as ft_mlp
from matplotlib import pyplot as plt
import seaborn as sns


data_columns_names = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concativity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_std",
    "texture_std",
    "perimeter_std",
    "area_std",
    "smoothness_std",
    "compactness_std",
    "concativity_std",
    "concave_points_std",
    "symmetry_std",
    "fractal_dimension_std",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concativity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]


# COLORS
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'


def pairplot(df: pd.DataFrame, features: list, target_col: str | None = None):
    """Draw pairplot for each features in the dataframe.

    Parameters:
      df (pd.DataFrame): Dataframe.
      features (list): List of numerical features name.
      targets (str) (optionnal): Target col name
    """
    # Change global plot param
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 8,
        'legend.markerscale': 2,
    })
    print(f"Drawing pairplot for {len(features)} features...", flush=True,
          end="")
    if target_col is not None and len(ft_mlp.get_class_list(
                                      df, target_col)) > 1:
        plot = sns.pairplot(
            df, hue=target_col, vars=features,
            plot_kws={'alpha': 0.5, 's': 5})
    else:
        plot = sns.pairplot(df[features],
                            plot_kws={'alpha': 0.5, 's': 5})
    for ax in plot.axes.flatten():
        if ax is not None and ax.get_ylabel():
            ax.yaxis.label.set_rotation(45)
            current_label = ax.get_ylabel()
            if len(current_label) > 4:
                ax.set_ylabel(current_label[:4] + '...')
    plt.subplots_adjust(hspace=0.7, wspace=0.7, left=0.1,
                        right=0.9, top=0.95, bottom=0.1)
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"{GREEN} Success{RESET}")


def heatmap(df: pd.DataFrame):
    """Plot a heatmap of the correlation matrix for DataFrame

    Parrameters:
      df (pandas.DataFrame): DataFrame to use for correlation matrix
    """
    corr_matrix = ft_mlp.correlation_matrix(df)
    corr_df = pd.DataFrame(corr_matrix, columns=data_columns_names[2:]
                           + ['diagnosis'])  # back to df
    ax = sns.heatmap(corr_df, cmap="Blues", yticklabels=corr_df.columns,
                     xticklabels=corr_df.columns)
    ax.set_position([0.2, 0.2, 0.58, 0.7])
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.show()


def main(ac: int, av: list):
    """Give basic detailles on data

    Parameters:
      ac (int) : Number of parameters
      av (list) : List of parameters
    """
    try:
        if ac != 2:
            raise Exception("Usage: python analyzer.py <dataset.csv>")
        # Read csv file
        df = pd.read_csv(av[1], header=None)
        df.columns = data_columns_names

        # Extract data and target
        raw_data = df.drop(["id", "diagnosis"], axis=1)
        raw_target = df["diagnosis"]

        # Standardize data
        standardized_data = ft_mlp.standardize_df(raw_data)

        # Convert target to numeric
        numeric_target = ft_mlp.convert_classes_to_nbr('M', raw_target)

        # Describe data
        ft_mlp.ft_describe(raw_data)

        mean_data = standardized_data[data_columns_names[2:12]]
        std_data = standardized_data[data_columns_names[12:22]]
        worst_data = standardized_data[data_columns_names[22:32]]
        pairplot(df, mean_data.columns.tolist(), "diagnosis")
        pairplot(df, std_data.columns.tolist(), "diagnosis")
        pairplot(df, worst_data.columns.tolist(), "diagnosis")

        # Correlation heatmap
        standardized_data['diagnosis'] = numeric_target
        heatmap(standardized_data)
    except Exception as e:
        print(f"{RED}Error{RESET}: {e}")


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
