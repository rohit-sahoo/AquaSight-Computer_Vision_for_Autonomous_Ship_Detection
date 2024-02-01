import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results(csv_path):
    # Load the CSV data into a pandas DataFrame
    df = pd.read_csv(csv_path)
    # Remove any leading or trailing whitespace from column names
    df.columns = df.columns.str.strip()

    # Initialize a 5x2 grid of plots with a total figure size
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))

    # Plot the columns using seaborn
    sns.lineplot(x='epoch', y='train/box_loss', data=df, ax=axs[0,0])
    sns.lineplot(x='epoch', y='train/cls_loss', data=df, ax=axs[0,1])
    sns.lineplot(x='epoch', y='train/dfl_loss', data=df, ax=axs[1,0])
    sns.lineplot(x='epoch', y='metrics/precision(B)', data=df, ax=axs[1,1])
    sns.lineplot(x='epoch', y='metrics/recall(B)', data=df, ax=axs[2,0])
    sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=df, ax=axs[2,1])
    sns.lineplot(x='epoch', y='metrics/mAP50-95(B)', data=df, ax=axs[3,0])
    sns.lineplot(x='epoch', y='val/box_loss', data=df, ax=axs[3,1])
    sns.lineplot(x='epoch', y='val/cls_loss', data=df, ax=axs[4,0])
    sns.lineplot(x='epoch', y='val/dfl_loss', data=df, ax=axs[4,1])

    # Set titles and axis labels for each subplot
    axs[0,0].set(title='Train Box Loss')
    axs[0,1].set(title='Train Class Loss')
    axs[1,0].set(title='Train DFL Loss')
    axs[1,1].set(title='Metrics Precision (B)')
    axs[2,0].set(title='Metrics Recall (B)')
    axs[2,1].set(title='Metrics mAP50 (B)')
    axs[3,0].set(title='Metrics mAP50-95 (B)')
    axs[3,1].set(title='Validation Box Loss')
    axs[4,0].set(title='Validation Class Loss')
    axs[4,1].set(title='Validation DFL Loss')

    # Add suptitle and adjust layout
    plt.suptitle('Training Metrics and Loss', fontsize=24)
    plt.subplots_adjust(top=0.95)  # Adjust top margin to make space for suptitle
    plt.tight_layout()  # Adjust spacing between subplots

    plt.show()
