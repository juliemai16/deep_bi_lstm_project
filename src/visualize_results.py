import os
import matplotlib.pyplot as plt

# Plot utility
def plot_graphs(history, string, save_dir=None):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{string}.png"))
    else:
        plt.show()
