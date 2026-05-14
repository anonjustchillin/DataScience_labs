import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import numpy as np
import stanza
import os

PROJECT_PATH = 'D:\\uni\\3курс\\Data_Science\\Data_science_labs\\lab4\\data'
SELECTED_POS = ['ADP', 'PART', 'DET', 'SCONJ', 'CCONJ', 'PUNCT', 'PRON']


def filter_data(data, filename):
    # stanza.download('uk', processors='tokenize,mwt,pos,lemma')

    nlp = stanza.Pipeline('uk', processors='tokenize,mwt,pos,lemma')

    text_arr =  data['Comments'].values.tolist()
    text = ". ".join(text_arr)
    doc = nlp(text)
    lemmas = [word.lemma for t in doc.iter_tokens() for word in t.words]
    pos = [word.upos for t in doc.iter_tokens() for word in t.words]

    tokens_data = pd.DataFrame(
        {
            'Word': lemmas,
            'POS': pos,
        }
    )

    tokens_data.drop(tokens_data[tokens_data.Word == '.'].index, inplace=True)

    tokens_data.to_csv(filename)
    return


def show_freq_plot(df, name, title='Word Count'):
    # func for cool gradient
    def truncate_colormap(cmap, min_val=0.0, max_val=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=min_val, b=max_val),
            cmap(np.linspace(min_val, max_val, n)))
        return new_cmap

    data = df['Word'].value_counts()
    print(data.head(20))

    x = data.head(15).index.tolist()
    y = data.head(15).values.tolist()

    fig, ax = plt.subplots(figsize=[12, 8])
    bars = ax.bar(x, y)
    plt.grid(True, alpha=0.3)
    y_min, y_max = ax.get_ylim()
    grad = np.atleast_2d(np.linspace(0, 1, 256)).T
    ax = bars[0].axes
    lim = ax.get_xlim() + ax.get_ylim()
    for bar in bars:
        bar.set_zorder(1)
        bar.set_facecolor("none")
        x, _ = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        c_map = truncate_colormap(plt.cm.plasma, min_val=0,
                                  max_val=(h - y_min) / (y_max - y_min))
        ax.imshow(grad, extent=[x, x + w, h, y_min], aspect="auto", zorder=0,
                  cmap=c_map)
    ax.axis(lim)
    plt.title(title+' '+name)
    plt.xticks(rotation=20)
    plt.savefig('freq_plot_'+name+'.jpg', dpi=300)
    plt.show()
    return


def show_countplot(df):
    categories = df['Category'].unique().tolist()
    sns.countplot(data=df, x='Category', hue='Category', hue_order=categories)
    plt.savefig('countplot.jpg', dpi=300)
    plt.show()


def get_result_analysis(filename, name):
    df = pd.read_csv(filename, index_col=0)

    show_countplot(df)

    categories = df['Category'].unique().tolist()
    for category in categories:
        filename_i = os.path.join(PROJECT_PATH, name + '_tokenized_'+category+'.csv')
        filter_data(df.loc[df['Category']==category], filename_i)

        data = pd.read_csv(filename_i, index_col=0)
        data.drop(data[data['POS'].isin(SELECTED_POS)].index, inplace=True)
        show_freq_plot(data, category)
        print()
    return