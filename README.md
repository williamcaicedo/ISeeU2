# ISeeU2: Visually Interpretable ICU mortality prediction using deep learning and free-text medical notes

* Pre-print: [ArXiV](https://www.sciencedirect.com/science/article/abs/pii/S1532046419301881)

**Note: This version of ISeeU has been tested with numpy 1.12.1, pandas 0.23.4, keras 2.2.4, deeplift 0.6.6.2, matplotlib 2.0.2 and tensorflow 1.9.0.**


A ConvNet trained on free-text medical notes from MIMIC-III for mortality prediction inside the Intensive Care Unit. It uses nursing notes taken from the first 48h of ICU stay to predict the probability of mortality of patients. ISeeU2 also generate note heatmaps to visualize word and note-fragement importance, offering interpretability and showing the rationale behind predictions.


ISeeU2 achieves 0.8629 AUROC when evaluated on MIMIC-III. More information is available in our [ArXiV pre-print](https://www.sciencedirect.com/science/article/abs/pii/S1532046419301881). It also can be installed from PyPi:
```unix
pip install iseeu2
```

