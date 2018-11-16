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
    plt.scatter(x, y, s=marker_sizes, color='black')

    # Format x-axis label angles
    fig.autofmt_xdate()

    # Raise the bottom of the figure to make space for the angled labels
    plt.subplots_adjust(bottom=0.23)

    # Show data
    if timeout is not None:
        timer.start()
    plt.show()


def plot_prediction(x, y, y_pred=None, timeout=None):
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
    plt.scatter(x, y, s=marker_sizes, color='black')
    if y_pred is not None:
        plt.plot(x, y_pred, color='blue', linewidth=3)

    # Format x-axis label angles
    fig.autofmt_xdate()

    # Raise the bottom of the figure to make space for the angled labels
    plt.subplots_adjust(bottom=0.23)

    # Show data
    if timeout is not None:
        timer.start()
    plt.show()


def close_event():
    """Timer calls this lambda function to close the plot window"""
    plt.close()  # Closes the window
