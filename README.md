# MLMarker

This repository contains the data and model for the MLMarker algorithm that identifies tissue-specific protein expression patterns.

The data folder contains the training and test data.
./data/tissue_predictor_notfiltered_healthy_nofluid_2206.csv: this is the original tissue predictor dataset prior to any filtering
./data/training_atlas_92%_10exp.csv: this is the training data after filtering for maximum 92% missingness and at least 10 tissue samples within the data. 
./data/binary_training_atlas_92%_10exp.csv: this is the same atlas but NSAF values are replaced by 1. 
The details of filtering are described in the notebook ./Tissue_predictors_filtering_training_application. ipynb

The models folder contains the output from the training in the notebook on both the NSAF and binary atlas. Models are saved as both pkl and joblib, features are exported as txt separated by ',\n'

The MLMarker_app folder contains all modules used throughout with:
database: design and completion of the database from ionbot results;
atlas: building the atlas from the database;
predictor atlas: converting the atlas to usable training format;
cell and tissue_predictors: training modules;
utils: other functions.
Most importantly: the mlmarker.py script contains the modules for the app in where the model predicts an unseen data (e.g. ./data/test_sample.csv) and the SHAP modules to interpret the predictions.
