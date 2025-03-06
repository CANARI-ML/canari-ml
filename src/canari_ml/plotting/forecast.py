import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .cli import PlottingNumpyArgParser

def plot_numpy_prediction(numpy_file: str) -> None:
    """Plots direct forecast prediction output (numpy file).

    Args:
        numpy_file: Path to the numpy file containing forecast predictions.
    """
    prediction = np.load(numpy_file)

    # Get dimensions (time, height, width, leadtime)
    time_steps = prediction.shape[0]
    leadtimes = prediction.shape[3]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Initial set-up
    selected_time = 0
    selected_leadtime = 0

    pred_slice = prediction[selected_time, :, :, selected_leadtime]

    img = ax.imshow(pred_slice, cmap="viridis")
    ax.set_title(f"Time {selected_time + 1}, Leadtime {selected_leadtime + 1}")
    plt.colorbar(img, ax=ax)

    # Only want to create sliders if more than 1 element
    create_time_slider = time_steps > 1
    create_leadtime_slider = leadtimes > 1

    divider = make_axes_locatable(ax)
    if create_time_slider:
        time_slider_ax = divider.append_axes("bottom", "10%", pad=0.25)
        time_slider = Slider(
            ax=time_slider_ax,
            label="Time",
            valmin=1,
            valmax=time_steps,
            valinit=selected_time,
            valstep=np.linspace(1, time_steps, num=time_steps),
            orientation="horizontal",
        )
    if create_leadtime_slider:
        leadtime_slider_ax = divider.append_axes("bottom", "10%", pad=0.25)
        leadtime_slider = Slider(
            ax=leadtime_slider_ax,
            label="Leadtime",
            valmin=1,
            valmax=leadtimes,
            valinit=selected_leadtime,
            valstep=np.linspace(1, leadtimes, num=leadtimes),
            orientation="horizontal",
        )

    def update(val):
        """Update function for the sliders."""
        current_time = time_slider.val if create_time_slider else selected_time + 1
        current_leadtime = (
            leadtime_slider.val if create_leadtime_slider else selected_leadtime + 1
        )

        current_time = int(round(current_time))
        current_leadtime = int(round(current_leadtime))

        # Validate selections (clipping to available indices)
        if current_time < 1:
            current_time = 1
        elif current_time > time_steps:
            current_time = time_steps
        if current_leadtime < 1:
            current_leadtime = 1
        elif current_leadtime > leadtimes:
            current_leadtime = leadtimes

        # Extract new slice and update figure
        pred_slice = prediction[current_time - 1, :, :, current_leadtime - 1]
        img.set_data(pred_slice)
        ax.set_title(f"Time {current_time}, Leadtime {current_leadtime}")

        # Redraw figure
        fig.canvas.draw_idle()

    # Register update function with sliders
    if create_time_slider:
        time_slider.on_changed(update)
    if create_leadtime_slider:
        leadtime_slider.on_changed(update)

    # Button to reset to defaults
    reset_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(reset_ax, "Reset")

    def reset(event):
        if create_time_slider:
            time_slider.reset()
        if create_leadtime_slider:
            leadtime_slider.reset()
        return

    reset_button.on_clicked(reset)

    plt.show()


def plot_numpy():
    """CLI entrypoint to plot a direct numpy prediction output"""
    args = PlottingNumpyArgParser().parse_args()
    plot_numpy_prediction(args.numpy_file)
