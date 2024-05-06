import pandas as pd
import matplotlib.pyplot as plt

def create_boxplot(lstm_df: pd.DataFrame, qlstm_df: pd.DataFrame, column: str, title: str):
    # Filter the DataFrames for each epoch
    lstm_do_1 = lstm_df[lstm_df['epochs'] == 1]
    qlstm_do_1 = qlstm_df[qlstm_df['epochs'] == 1]

    lstm_do_5 = lstm_df[lstm_df['epochs'] == 5]
    qlstm_do_5 = qlstm_df[qlstm_df['epochs'] == 5]

    lstm_do_15 = lstm_df[lstm_df['epochs'] == 15]
    qlstm_do_15 = qlstm_df[qlstm_df['epochs'] == 15]

    lstm_do_30 = lstm_df[lstm_df['epochs'] == 30]
    qlstm_do_30 = qlstm_df[qlstm_df['epochs'] == 30]

    lstm_do_50 = lstm_df[lstm_df['epochs'] == 50]
    qlstm_do_50 = qlstm_df[qlstm_df['epochs'] == 50]

    # Combine the filtered "mse_train" values for each epoch
    combined_epochs_1 = [lstm_do_1[column], qlstm_do_1[column]]
    combined_epochs_5 = [lstm_do_5[column], qlstm_do_5[column]]
    combined_epochs_15 = [lstm_do_15[column], qlstm_do_15[column]]
    combined_epochs_30 = [lstm_do_30[column], qlstm_do_30[column]]
    combined_epochs_50 = [lstm_do_50[column], qlstm_do_50[column]]


    # Create the subplot grid with two rows and three columns
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    # Define custom colors for LSTM and QLSTM boxes
    lstm_color = "#1f77b4"
    qlstm_color = "orange"

    # create boxplots with custom colors

    bp_epochs_1 = axes[0, 0].boxplot(
        combined_epochs_1,
        labels=['LSTM', 'QLSTM'],
        patch_artist=True)
    for box, color in zip(bp_epochs_1['boxes'], [lstm_color, qlstm_color]):
        box.set_facecolor(color)

    bp_epochs_5 = axes[0, 1].boxplot(
        combined_epochs_5,
        labels=['LSTM', 'QLSTM'],
        patch_artist=True)
    for box, color in zip(bp_epochs_5['boxes'], [lstm_color, qlstm_color]):
        box.set_facecolor(color)

    bp_epochs_15 = axes[0, 2].boxplot(
        combined_epochs_15,
        labels=['LSTM', 'QLSTM'],
        patch_artist=True)
    for box, color in zip(bp_epochs_15['boxes'], [lstm_color, qlstm_color]):
        box.set_facecolor(color)

    bp_epochs_30 = axes[1, 0].boxplot(
        combined_epochs_30,
        labels=['LSTM', 'QLSTM'],
        patch_artist=True)
    for box, color in zip(bp_epochs_30['boxes'], [lstm_color, qlstm_color]):
        box.set_facecolor(color)

    bp_epochs_50 = axes[1, 1].boxplot(
        combined_epochs_50,
        labels=['LSTM', 'QLSTM'],
        patch_artist=True)
    for box, color in zip(bp_epochs_50['boxes'], [lstm_color, qlstm_color]):
        box.set_facecolor(color)


    # Hide the x-axis labels for the first row
    axes[0, 0].set_xticklabels([])
    axes[0, 1].set_xticklabels([])
    axes[0, 2].set_xticklabels([])

    # Optional: Add labels and title for each subplot
    axes[0, 0].set_ylabel('Mean Squared Error (MSE) - Train')
    axes[0, 0].set_title('Epoch 1')
    axes[0, 1].set_title('Epoch 5')
    axes[0, 2].set_title('Epoch 15')

    axes[1, 0].set_ylabel('Mean Squared Error (MSE) - Train')
    axes[1, 0].set_title('Epoch 30')
    axes[1, 1].set_title('Epoch 50')

    # Hide the empty subplots
    axes[1, 2].axis('off')

    fig.suptitle(title, fontsize=16)

    # Show the plot
    plt.tight_layout()
    plt.show()



def create_epoch_plot(lstm_df: pd.DataFrame, qlstm_df: pd.DataFrame, title: str):
    lstm_custom_data_epochs_mean = lstm_df.select_dtypes(include='number').mean() \
                                        .drop(["epochs", "mse_train", "mse_test", "mae_train", "mae_test", "Reps"])

    qlstm_custom_data_epochs_mean = qlstm_df.select_dtypes(include='number').mean() \
                                            .drop(["epochs", "mse_train", "mse_test", "mae_train", "mae_test", "Reps"])

    # plot the metrics
    lstm_custom_data_epochs_mean.plot(label="LSTM")
    qlstm_custom_data_epochs_mean.plot(label="QLSTM")

    # meta data for the plot
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()

    # customize the x-ticks
    x_ticks = range(0, 51, 5)  
    plt.xticks(x_ticks, labels=x_ticks)

    plt.show()