import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

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






def plot_Comparative_sentiment_distribution_between_cops(cop26_grouped, cop27_grouped, title='title'):
    

    # Aggregate totals for each sentiment category
    sentiment_totals_cop26 = cop26_grouped[['Positive', 'Neutral', 'Negative']].sum()
    sentiment_totals_cop27 = cop27_grouped[['Positive', 'Neutral', 'Negative']].sum()

    # Convert to percentages
    sentiment_percentages_cop26 = sentiment_totals_cop26 / sentiment_totals_cop26.sum() * 100
    sentiment_percentages_cop27 = sentiment_totals_cop27 / sentiment_totals_cop27.sum() * 100

    # Create DataFrame
    sentiment_percentages = pd.DataFrame({
        'COP26': sentiment_percentages_cop26,
        'COP27': sentiment_percentages_cop27
    })


    # Define the colors based on sentiment
    sentiment_colors = {
        'Positive': 'green',
        'Neutral': '#bfbfbf',  # Lighter gray for better visibility
        'Negative': 'red'
    }

    # Define hatch patterns for COP26 and COP27
    hatch_patterns = {
        'COP26': '',        # No hatch for COP26
        'COP27': '///'      # Diagonal hatching for COP27
    }

    # Plot the grouped bar chart
    ax = sentiment_percentages.plot(kind='bar', figsize=(10, 6), color=['white', 'white'], edgecolor='black')

    # Get the sentiment categories and conferences
    sentiments = sentiment_percentages.index.tolist()     # ['Positive', 'Neutral', 'Negative']
    conferences = sentiment_percentages.columns.tolist()  # ['COP26', 'COP27']

    # Customize bars
    for i, bar_container in enumerate(ax.containers):
        conference = conferences[i]
        hatch = hatch_patterns[conference]
        for j, bar in enumerate(bar_container):
            sentiment = sentiments[j]
            color = sentiment_colors[sentiment]
            bar.set_facecolor(color)
            bar.set_hatch(hatch)
            bar.set_edgecolor('black')  # Ensures the hatch patterns are visible

    # Remove x-axis label
    ax.set_xlabel('')

    # Set y-axis label
    ax.set_ylabel('Percentage of Tweets (%)', fontsize=12)

    # Set the x-tick labels to be horizontal
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # Get the figure object
    fig = ax.get_figure()

    # Adjust the bottom margin to make room for the title
    fig.subplots_adjust(bottom=0.8)  # Increase bottom margin to 20% of the figure height

    # Add title at the bottom
    fig.text(0.5, 0.01, title, ha='center', fontsize=16)

    # Create custom legend handles for COP26 and COP27 only
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', hatch='', label='COP26'),
        Patch(facecolor='white', edgecolor='black', hatch='///', label='COP27')
    ]

    # Add the legend to the plot
    ax.legend(handles=legend_elements, title='Conference')

    # Optional: Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 1])  # Adjust the rect parameter to prevent tight_layout from overlapping with subplots_adjust

    # Show the plot
    plt.show()







def active_engos(df_cop_relevance, title='title'):
    # Calculate average and standard deviation for tweet counts
    average_tweets_cop = df_cop_relevance.groupby('Name')['label'].count().mean()
    std_dev_tweets_cop = df_cop_relevance.groupby('Name')['label'].count().std()

    # Identify active NGOs
    threshold = average_tweets_cop + std_dev_tweets_cop
    active_ngos_cop= df_cop_relevance.groupby('Name').filter(lambda x: len(x) > threshold)

    # Aggregate counts of each label for selected NGOs
    active_ngos_summary_cop= active_ngos_cop.groupby(['Name', 'label'])['label'].count().unstack(fill_value=0).reset_index()
    active_ngos_summary_cop['Total'] = active_ngos_summary_cop['positive'] + active_ngos_summary_cop['negative']

    # Sort NGOs by total number of tweets
    # sorted_active_ngos_summary_cop = active_ngos_summary_cop.sort_values(by='Total', ascending=False)

    # The 'sorted_active_ngos_summary' dataframe is now ready for plotting the stacked bar graph

    # Ensure the data is sorted by the total number of tweets if not already sorted
    active_ngos_summary_sorted_cop = active_ngos_summary_cop.sort_values(by='Total', ascending=False)
    active_ngos_summary_sorted_cop['difference'] = active_ngos_summary_sorted_cop['negative'] - active_ngos_summary_sorted_cop['positive']
    group_info_cop = df_cop_relevance[['Name', 'Group']].drop_duplicates()
    # Merge the 'Group' information into the 'active_ngos_summary_sorted' dataframe
    active_ngos_summary_sorted_cop = pd.merge(active_ngos_summary_sorted_cop, group_info_cop, on='Name', how='left')



    import matplotlib.pyplot as plt

    def plot_horizontal_stacked_bar(data, title):
        fig, ax = plt.subplots(figsize=(10, 12))

        data = data.iloc[::-1]

        ax.barh(data['Name'],
                data['positive'],
                label='Positive',
                color='green')

        ax.barh(data['Name'],
                data['negative'],
                left=data['positive'],
                label='Negative',
                color='red')

        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

        ax.set_xlabel('Number of Tweets')
        ax.set_ylabel('')

        # Adjust legend position: bbox_to_anchor=(x, y)
        # x: 0 is left edge, 1 is right edge
        # y: 0 is bottom, 1 is top
        ax.legend(loc='center right', bbox_to_anchor=(0.8, 0.05))

        fig.text(0.5, 0.02, title, ha='center', va='center', fontsize=20)

        plt.subplots_adjust(left=0.4, top=0.95)
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        return fig, ax

    fig, ax = plot_horizontal_stacked_bar(
        active_ngos_summary_sorted_cop,
        title
    )
    plt.show()
        



