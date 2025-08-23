from setuptools import setup, find_packages

setup(
    name="mlmarker",
    version="0.1.6",
    description="MLMarker is a Python package for tissue-specific protein marker prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Tine Claeys",
    author_email="tineclae.claeys@ugent.be",
    packages=find_packages(include=["mlmarker", "mlmarker.*"]),
    include_package_data=True,
    package_data={
        "mlmarker": ["models/*.joblib", "models/*.txt"]
    },
    install_requires=[
        "numpy==1.23.5",  # Pin to numpy 1.x
        "pandas>=2.0.0",
        "scikit-learn>=1.0.0",
        "shap>=0.41.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "gprofiler-official>=1.0.0",
    ],
    python_requires=">=3.8,<3.13",  # Restrict Python version
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    }
)
