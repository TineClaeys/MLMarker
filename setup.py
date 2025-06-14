from setuptools import setup, find_packages

setup(
    name='mlmarker',
    version='0.1.5',
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
        'numpy==1.26.4',
        'bioservices',
        "shap==0.42.0"
    ],
)
