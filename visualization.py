import matplotlib.pyplot as plt

from wordcloud import WordCloud

def boxplot(x_data, y_data, base_color="#2196F3", median_color="#FF5252", x_label="", y_label="", title=""):
    _, ax = plt.subplots()

    # Draw boxplots, specifying desired style
    ax.boxplot(y_data
               # patch_artist must be True to control box fill
               , patch_artist = True
               # Properties of median line
               , medianprops = {'color': median_color}
               # Properties of box
               , boxprops = {'color': base_color, 'facecolor': base_color}
               # Properties of whiskers
               , whiskerprops = {'color': median_color}
               # Properties of whisker caps
               , capprops = {'color': base_color}
               , flierprops = {'markerfacecolor':'#BBDEFB'}
               )

    # By default, the tick label starts at 1 and increments by 1 for
    # each box drawn. This sets the labels to the ones we want
    ax.set_xticklabels(x_data)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    plt.show()


def wordcloud(freq, title = None):
    wordcloud = WordCloud(
        background_color='white',
        max_words=150,
        max_font_size=40,
        scale=3,
        colormap="Set2",
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate_from_frequencies(freq)

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
