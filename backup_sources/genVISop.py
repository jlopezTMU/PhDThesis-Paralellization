## imports for Optuna
import os
import optuna
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend which is GUI-less
import matplotlib.pyplot as plt
import numpy as np

from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_contour
#
def main():

    study = optuna.create_study(
    direction='maximize',
    study_name='myresearch_study',  # Name your study
    storage='postgresql://jlopez:dbpass@localhost/optuna_study',  # Use your PostgreSQL credentials and database name
    load_if_exists=True)

 # Ensure you have a headless backend for Matplotlib if you're on a server without GUI
    plt.switch_backend('agg')

    # Plotting optimization history
    ax = plot_optimization_history(study)
    fig = ax.figure
    fig.set_size_inches(8, 6)  # Set the dimensions after getting the figure
    fig.savefig("optimization_history.png")

    # Plotting parameter importances
    ax = plot_param_importances(study)
    fig = ax.figure
    fig.set_size_inches(9, 7)  # Set the dimensions after getting the figure
    fig.savefig("param_importances.png")

    # Plotting contour plot
    axes_array = plot_contour(study)
    fig = axes_array.ravel()[0].figure  # Access the figure from one of the subplots
    fig.set_size_inches(8, 6)  # Set the dimensions after getting the figure
    fig.savefig("plot_contour.png")

main()
