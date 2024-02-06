import os
import pandas as pd
import matplotlib.pyplot as plt
directory = "./"

for filename in os.listdir(directory):
    if filename.endswith(".csv") and 'E5' in filename:
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)
        ncols = df.shape[1]
        fig, axs = plt.subplots(ncols, 1, figsize=(10, 10))

        # Plot each column in a separate subplot
        for i in range(ncols):
            axs[i].plot(df.iloc[:, i])
            axs[i].set_title(filename)

        # Show the plot
        plt.show()