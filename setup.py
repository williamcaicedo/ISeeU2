from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    include_package_data=True,
    name='iseeu2',
    packages=['iseeu2'],
    version='0.1.0',
    license='MIT',
    description='ISeeU2: Visually Interpretable ICU mortality prediction using deep learning and free-text medical notes',
    long_description='''A ConvNet trained on free-text medical notes from MIMIC-III for mortality prediction inside the Intensive Care Unit. It uses nursing notes taken from the first 48h of ICU stay to predict the probability of mortality of patients. ISeeU2 also generate note heatmaps to visualize word and note-fragement importance, offering interpretability and showing the rationale behind predictions.
    ''',
    author='William Caicedo-Torres',
    url='https://github.com/williamcaicedo/ISeeU2',
    keywords=['Deep Learning', 'Mortality prediction', 'Shapley values'],
    install_requires=[
        'numpy>=1.18.4',
        'scipy>=1.4.1',
        'pandas>=1.0.3',
        'shap>=0.35.0',
        'matplotlib>=3.2.1',
        'tensorflow>=2.1.0',
        'nltk>=3.2.5',
        'wordcloud>=1.5.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
    ],
    zip_safe=False
)
