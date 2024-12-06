from setuptools import setup, find_packages

setup(
    name='MLMarker_app',
    version='0.1',
    description="MLMarker is a Python package for tissue-specific protein marker prediction",
    author="Tine Claeys",
    author_email="tineclae.claeys@ugent.be",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "mlmarker": ["models/*.joblib", "models/*.txt"]
    },
    install_requires=[
        'pandas',
        'scikit-learn',
        'matplotlib',
        'shap',
        'seaborn',
        'numpy'
    ],
)
