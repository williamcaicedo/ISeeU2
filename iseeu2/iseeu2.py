import pickle
import pkg_resources
from wordcloud import WordCloud
import shap
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

import tensorflow
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.compat.v1.keras.backend import get_session
tensorflow.compat.v1.disable_v2_behavior()


class ISeeU2:
    __version__ = "0.1.0"

    def __init__(self):
        model_file = pkg_resources.resource_filename(
            'iseeu2', 'models/kfold2_best_F1.h5')
        print(f"****{model_file}*****")
        self._model = load_model(model_file)

        self._stops = set(stopwords.words("english"))
        tokenizer_file = pkg_resources.resource_filename(
            'iseeu2', 'res/tokenizer.pkl')
        
        with open(tokenizer_file, 'rb') as t:
            self._tokenizer = pickle.load(t)
        self._sequence_max_length = 500
        background_file = pkg_resources.resource_filename(
            'iseeu2', 'res/explainer_background.pkl')
        
        with open(background_file, 'rb') as b:
            background = pickle.load(b)
        self._explainer = shap.DeepExplainer(self._model, background)

        word_index = self._tokenizer.word_index
        self._reverse_word_index = dict(
            [(value, key) for (key, value) in word_index.items()])

    def preprocess_notes(self, notes):
        def f(x): return ' '.join(
            [item for item in x.split() if item not in self._stops])
        notes['text'] = notes['text'].apply(f)
        notes_sequences = self._tokenizer.texts_to_sequences(
            notes['text'].values)
        padded_notes_sequences = pad_sequences(notes_sequences,
                                               self._sequence_max_length,
                                               truncating='pre')
        return padded_notes_sequences

    def predict(self, patient_sequences):
        mortality = self._model.predict(patient_sequences)
        shapley_values = self._explainer.shap_values(patient_sequences)
        return mortality, shapley_values

    def _get_word_importance(self, patient_note_sequence, shapley_values):
        words = [self._reverse_word_index.get(
            i, '*') for i in patient_note_sequence]
        death_words_importance = {
            k: v for (k, v) in zip(words, shapley_values) if v > 0}
        survival_words_importance = {
            k: v*-1 for (k, v) in zip(words, shapley_values) if v < 0}
        return death_words_importance, survival_words_importance

    def _get_viz_color(self, scores, convolve=False):
        cmap='coolwarm'
        norm = MidpointNormalize(vmin=scores.min(), vmax=scores.max(), midpoint=0)
        if convolve:
            scores = convolve1d(scores, weights=[0.1, 0.2, 0.4, 0.2, 0.1])
        scaled_scores = norm(scores)
        heatmap_cm = cm.get_cmap(cmap)
        heatmap_colors = heatmap_cm(scaled_scores)
        return heatmap_colors
    
    def _create_note_importance_viz(self, words, heatmap_colors):
        text = "<p style='font-size:17px;width:1000px'>"
        for w, c in zip(words, heatmap_colors):
            text += f"<span style='background-color: rgba({','.join([str(int(v*255)) for v in c[:-1]])}, 0.7)'>{w}</span> "
        text += '</p>'
        return text

    def get_note_heatmap(self, patient_note_sequence, shapley_values, convolve = False):
        words = [self._reverse_word_index.get(
            i, '*') for i in patient_note_sequence]
        heatmap_colors = self._get_viz_color(shapley_values, convolve=convolve)
        html = self._create_note_importance_viz(words, heatmap_colors)
        return html

    def get_word_clouds(self, patient_note_sequence, shapley_values, filename = None):
        death_words_importance, survival_words_importance = self._get_word_importance(patient_note_sequence,
                                                                                      shapley_values)
        wc_death = WordCloud(background_color="white", max_words=100,
                             width=1500, height=1500, colormap='cubehelix')
        wc_survival = WordCloud(background_color="white", max_words=100,
                                width=1500, height=1500, colormap='cubehelix')
        # generate word cloud
        wc_death.generate_from_frequencies(death_words_importance)
        wc_survival.generate_from_frequencies(survival_words_importance)
        
        fig, ax = plt.subplots(1, 2, figsize=(25,15))
        ax[0].imshow(wc_survival, interpolation="bilinear")
        ax[1].imshow(wc_death, interpolation="bilinear")
        ax[0].axis("off") 
        ax[1].axis("off")
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=500)
        plt.show()

# set the colormap and centre the colorbar
# http://chris35wills.github.io/matplotlib_diverging_colorbar/


class MidpointNormalize(mpl.colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
