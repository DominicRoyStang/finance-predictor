import matplotlib.pyplot as plt
import numpy


def plot_data(x, y, timeout=None):
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
    plt.scatter(y, x, s=marker_sizes, color='black')
    # Set the number of ticks on the x-axis
    xmin, xmax = ax.get_xlim()
    ax.set_xticks(numpy.linspace(xmin, xmax, 9))
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
