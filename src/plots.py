import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sentiment_comparison3(data_sets, sentiment_type="negative", date_column="date", legend_names=None, events=None, moving_avg_window=None, title='title'):
    # Set seaborn style for better aesthetics
    sns.set_theme(style="whitegrid")

    # Convert date column to datetime object and calculate moving average if needed
    for data_set in data_sets:
        data_set[date_column] = pd.to_datetime(data_set[date_column])
        if moving_avg_window and moving_avg_window > 0:
            data_set[f'{sentiment_type}_moving_avg'] = data_set[sentiment_type].rolling(window=moving_avg_window, min_periods=1, center=True).mean()

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 6))
    for idx, data_set in enumerate(data_sets):
        if moving_avg_window and moving_avg_window > 0:
            sentiment_data = data_set[f'{sentiment_type}_moving_avg']
        else:
            sentiment_data = data_set[sentiment_type]

        ax.plot(data_set[date_column], sentiment_data,
                label=legend_names[idx] if legend_names else f'Data Set {idx + 1}',
                color=sns.color_palette("Set2")[idx],
                linestyle='-', linewidth=2, marker='o')

    # Add dotted lines and text descriptions for events
    if events:
        for event_date, event_label in events.items():
            event_date = pd.to_datetime(event_date)
            ax.axvline(x=event_date, linestyle='--', color='gray', linewidth=1)
            ax.text(event_date, ax.get_ylim()[1], f' {event_label}', rotation=0, verticalalignment='bottom', ha='center', fontsize=14, color='black')

    plt.xlabel('Date', fontsize=14, fontweight='bold')
    plt.ylabel('Sentiment Index', fontsize=16, fontweight='bold')

    # Manually adding a title below the plot
    fig.text(0.5, -0.07, title, ha='center', va='center', fontsize=24)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    plt.grid(True)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.show()

# Example usage (assuming data_sets, legend_names, and events are defined):
# plot_sentiment_comparison3(data_sets, sentiment_type='negative', date_column='date', legend_names=legend_names, events=events, moving_avg_window=7)
