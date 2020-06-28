# ISeeU2: Visually Interpretable ICU mortality prediction using deep learning and free-text medical notes

* Pre-print: [ArXiV](https://arxiv.org/abs/2005.09284)

**Note: This version of ISeeU2 has been tested with numpy 1.18.4, pandas 1.0.3, scipy 1.4.1, tensorflow 2.1.0, shap 0.35.0, matplotlib 3.2.1, NLTK 3.2.5, and wordcloud 1.5.0.**


A ConvNet trained on free-text medical notes from MIMIC-III for mortality prediction inside the Intensive Care Unit. It uses nursing notes taken from the first 48h of ICU stay, to predict the probability of mortality of patients. ISeeU2 also generate note heatmaps to visualize word and note-fragement importance, offering interpretability and showing the rationale behind predictions.


ISeeU2 achieves 0.8629 AUROC when evaluated on MIMIC-III. It also can be installed from PyPi:
```unix
pip install iseeu2
```

