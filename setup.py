from setuptools import setup, find_packages

setup(
    name='mlmarker',
    version='0.1.1',
    description="MLMarker is a Python package for tissue-specific protein marker prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Tine Claeys",
    author_email="tineclae.claeys@ugent.be",
    packages=find_packages(include=["mlmarker", "mlmarker.*"]),  # Include only relevant package
    include_package_data=True,  # Respect MANIFEST.in
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
