import matplotlib.pyplot as plt
import numpy


def plot_data(x, y, timeout=None):
    """Draws a scatter plot"""

    # Resize the graph window
    fig = plt.figure(figsize=(10, 7))

    # Auto-close after timeout if one is specified
    if timeout is not None:
        timer = fig.canvas.new_timer(interval=timeout)
        timer.add_callback(close_event)

    # Get the axes
    ax = plt.axes()

    # Show the grid
    ax.grid()

    # Select marker size
    marker_sizes = [12 for _ in x]

    # Plot outputs
    plt.scatter(x, y, s=marker_sizes, color='darkorange')

    # Format x-axis label angles
    fig.autofmt_xdate()

    # Raise the bottom of the figure to make space for the angled labels
    plt.subplots_adjust(bottom=0.23)

    # Show data
    if timeout is not None:
        timer.start()
    plt.show()


def plot_prediction(x, y, X_test=None, y_pred=None, timeout=None):
    """Draws the data as a scatter plot, and the prediction as a line"""

    # Resize the graph window
    fig = plt.figure(figsize=(10, 7))

    # Auto-close after timeout if one is specified
    if timeout is not None:
        timer = fig.canvas.new_timer(interval=timeout)
        timer.add_callback(close_event)

    # Get the axes
    ax = plt.axes()

    # Show the grid
    ax.grid()

    # Select marker size
    marker_sizes = [12 for _ in x]

    # Plot outputs
    plt.scatter(x, y, s=marker_sizes, color='darkorange')
    if y_pred is not None:
        plt.plot(x, y_pred, color='navy', linewidth=1)

    # Draw vertical line at start of test set
    if X_test is not None:
        plt.axvline(x=X_test[0], linestyle='dashed', color='pink')

    # Format x-axis label angles
    fig.autofmt_xdate()

    # Raise the bottom of the figure to make space for the angled labels
    plt.subplots_adjust(bottom=0.23)

    # Show data
    if timeout is not None:
        timer.start()
    plt.show()


def plot_predictions(x, y, pred_dates=None, lines=None):
    """Draws the data as a scatter plot, and the predictions as lines"""

    # Resize the graph window
    fig = plt.figure(figsize=(10, 7))

    # Get the axes
    ax = plt.axes()

    # Show the grid
    # ax.grid()

    # Select marker size
    marker_sizes = [4 for _ in x]

    # Plot outputs
    if not(lines is None or pred_dates is None):
        plt.axvline(x=x[-1], linestyle='dashed', color='darkgray')
        for line, color in zip(lines, ['goldenrod', 'green', 'red']):
            plt.plot(pred_dates, line, color=color, linewidth=3, alpha=0.8)
    plt.scatter(x, y, s=marker_sizes, color='darkgray', zorder=100)

    # Format x-axis label angles
    fig.autofmt_xdate()

    # Raise the bottom of the figure to make space for the angled labels
    plt.subplots_adjust(bottom=0.23)

    plt.show()


def close_event():
    """Timer calls this lambda function to close the plot window"""
    plt.close()  # Closes the window
