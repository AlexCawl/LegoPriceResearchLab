import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def regression_stage_visualization(actual: pd.DataFrame, expected: pd.DataFrame, path: str, name: str) -> None:
    # setup scatter data
    scatter_data: pd.DataFrame = pd.DataFrame()
    scatter_data['Actual'] = actual
    scatter_data['Expected'] = expected
    # setup difference data
    diff_data: pd.DataFrame = pd.DataFrame()
    diff_data['Actual'] = actual
    diff_data['Difference'] = actual - expected
    # reload painter
    plt.clf()
    _, axs = plt.subplots(ncols=2, figsize=(16, 8))
    sns.regplot(data=scatter_data, x='Actual', y='Expected', ax=axs[0])
    sns.scatterplot(data=diff_data, x='Actual', y='Difference', ax=axs[1])
    plt.savefig(f"{path}/{name}.png")
    sns.reset_orig()
    plt.clf()


def regression_visualization(analytics: pd.DataFrame, path: str, name: str) -> None:
    plt.clf()
    _, axs = plt.subplots(figsize=(8, 8))
    sns.scatterplot(data=analytics, x='Actual', y='Expected', hue='Theme name', ax=axs)
    plt.savefig(f"{path}/{name}.png")
    sns.reset_orig()
    plt.clf()


def regressions_scores(analytics: pd.DataFrame, path: str, name: str) -> None:
    plt.clf()
    _, axs = plt.subplots(nrows=3, figsize=(8, 12))
    # MAE plot
    sns.lineplot(data=analytics, x='Theme name', y='MAE', ax=axs[0])
    # MSE plot
    sns.lineplot(data=analytics, x='Theme name', y='MSE', ax=axs[1])
    # RMSE plot
    sns.lineplot(data=analytics, x='Theme name', y='RMSE', ax=axs[2])
    plt.savefig(f"{path}/{name}-errors.png")
    plt.clf()
    # R2 plot
    _, axs = plt.subplots(figsize=(12, 8))
    sns.lineplot(data=analytics, x='Theme name', y='R2', hue='type')
    plt.savefig(f"{path}/{name}-r2.png")
    plt.clf()
