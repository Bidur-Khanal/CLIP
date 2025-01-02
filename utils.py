import torch
import matplotlib.pyplot as plt

class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
    
class FastDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

def plot_per_sample_losses(per_sample_losses, title="Per-Sample Losses", xlabel="Sample Index", ylabel="Loss"):
    """
    Plots per-sample losses as a scatter plot.

    Parameters:
        per_sample_losses (list or array): A list or array of per-sample losses.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.

    Returns:
        matplotlib.figure.Figure: The created figure object.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(range(len(per_sample_losses)), per_sample_losses, alpha=0.6, edgecolors='w', s=40)
    
    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Return the figure
    return fig


            
