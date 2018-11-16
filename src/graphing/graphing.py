import matplotlib.pyplot as plt
import matplotlib.dates
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
    # Change label angles
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)
    # Raise the bottom of the figure to make space for the angled labels
    plt.subplots_adjust(bottom=0.23)
    # Show data
    if timeout is not None:
        timer.start()
    plt.show()


def close_event():
    plt.close()  # timer calls this function after 3 seconds and closes the window
