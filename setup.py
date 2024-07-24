from setuptools import setup, find_packages

setup(
    name='MLMarker_app',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'mysql-connector-python',
        'imbalanced-learn',
        'scikit-learn',
        'matplotlib',
        'shap',
        'seaborn',
        'xgboost',
        'numpy'
    ],
)
