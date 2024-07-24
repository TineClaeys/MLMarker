import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

class MLMarker:
    def __init__(self, sample_df, binary=True):
        if binary:
            model_path = "/home/compomics/git/MLMarker/models/binary_TP_full_92%_10exp_2024.joblib"
            features_path = "/home/compomics/git/MLMarker/models/binary_features_TP_full_92%_10exp_2024.txt"
        else:
            model_path = "/home/compomics/git/MLMarker/models/TP_full_92%_10exp_2024.joblib"
            features_path = "/home/compomics/git/MLMarker/models/features_TP_full_92%_10exp_2024.txt"
        self.model_path = model_path
        self.features_path = features_path
        self.model, self.features = self.load_model_and_features(model_path, features_path)
        self.sample = self.read_sample(sample_df)
        self.binary = binary
        self.explainer = shap.TreeExplainer(self.model)  # Load SHAP explainer once

    def load_model_and_features(self, model_path, features_path):
        self.model = joblib.load(self.model_path)
        with open(features_path, 'r') as features_file:
            self.features = features_file.read().split(',\n')
        return self.model, self.features

    def read_sample(self, sample_df):
        # sample should be one row only! otherwise give error
        if sample_df.shape[0] > 1:
            print("Error: Sample should be a single row for prediction, only first row is used")
        # match features between sample column names and self.features
        matched_features = [feature for feature in self.features if feature in sample_df.columns]
        added_features = [feature for feature in self.features if feature not in sample_df.columns]
        removed_features = [feature for feature in sample_df.columns if feature not in self.features]
        if len(added_features) > 0:
            print("Warning: {} model features are not in the sample and were added as zero values".format(len(added_features)))
        if len(removed_features) > 0:
            print("Warning: {} sample features are not in the model and were removed".format(len(removed_features)))
        
        # the final sample contains matched and added features
        sample_df = sample_df[matched_features]
        added_features_df = pd.DataFrame(0, index=sample_df.index, columns=added_features)
        sample = pd.concat([sample_df, added_features_df], axis=1)
        return sample

    def predict_top_tissues(self, n_preds=5):
        probabilities = self.model.predict_proba(self.sample).flatten()
        classes = self.model.classes_
        result = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)[:n_preds]
        formatted_result = [(pred_tissue, round(float(prob), 4)) for pred_tissue, prob in result]
        return formatted_result

    def calculate_shap(self):
        shap_values = self.explainer.shap_values(self.sample, check_additivity=False)
        shap_values = np.transpose(shap_values, (0, 2, 1))[0]
        return shap_values

    def visualize_shap_force_plot(self, n_preds=5):
        shap_values = self.calculate_shap()
        predictions = self.predict_top_tissues(n_preds)
        shap.initjs()
        # print the base_value
        print("The base value is {}".format(self.explainer.expected_value[1]))
        for tissue, _ in predictions:
            tissue_loc = list(self.model.classes_).index(tissue)
            print(tissue)
            display(shap.force_plot(self.explainer.expected_value[1], shap_values[tissue_loc], self.sample, matplotlib=True))

    def interpret_shap_values(self, n_preds=5):
        """Get a dataframe with the SHAP values for each feature for the top n_preds tissues"""
        shap_values = self.calculate_shap(self.sample)
        classes = self.model.classes_
        predictions = self.predict_top_tissues(self.sample, n_preds)
        
        shap_values_df = pd.DataFrame(shap_values)
        shap_values_df.columns = self.features
        shap_values_df['tissue'] = classes
        shap_values_df = shap_values_df.set_index('tissue')
        shap_values_df = shap_values_df.loc[[item[0] for item in predictions]]
        return shap_values_df

    def pie_chart_predictions(self):
        predictions = self.model.predict_proba(self.sample).flatten()
        classes = self.model.classes_
        result_dict = {classes[i]: predictions[i] for i in range(len(classes))}
        sorted_tissues = sorted(result_dict.items(), key=lambda pair: -pair[1])
        
        tissue_names = [item[0] for item in sorted_tissues]
        tissue_predictions = [item[1] for item in sorted_tissues]
        
        plt.figure(figsize=(10, 6))
        plt.pie(tissue_predictions, labels=tissue_names, autopct=lambda p: '{:.1f}%'.format(p) if p > 3.4 else '')
        plt.show()

    def shap_abundance_distribution(self, n_preds=5):
        """For each classification, make a scatterplot of the SHAP values versus the abundance within the sample"""
        shap_values = self.calculate_shap()
        predictions = self.predict_top_tissues(n_preds)
        classes = self.model.classes_

        for tissue, _ in predictions:
            tissue_loc = list(classes).index(tissue)
            tissue_shap = shap_values[tissue_loc]
            plt.figure(figsize=(10, 6))
            plt.scatter(self.sample, tissue_shap, c=['black' if abundance == 0 else 'blue' for abundance in self.sample.values.flatten()])
            plt.xlabel("Abundance")
            plt.ylabel("SHAP value")
            plt.title("Abundance vs SHAP for {}".format(tissue))
            plt.show()
    
    def training_instances(self, n_preds=5):
        """Return the number of training instances for the top predictions"""
        training_instances = pd.read_csv('/home/compomics/git/MLMarker/data/training_instances.csv')
        predictions = self.predict_top_tissues(n_preds)
        pred_tissues = [tup[0] for tup in predictions]
        training_instances = training_instances.set_index('tissue_name')
        training_instances = training_instances.loc[pred_tissues]
        return training_instances.reset_index()