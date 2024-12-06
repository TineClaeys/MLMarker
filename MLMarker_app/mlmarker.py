import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import xgboost as xgb
import plotly.express as px

# Save the model as joblib file and save the feature names
from joblib import dump
import numpy as np

class MLMarker:
    def __init__(self, sample_df, binary=True, minimal_feature_set=False, dev=False):
        if binary:
            model_path = "/home/compomics/git/MLMarker/models/binary_TP_4000features_95to75missingness_2024.joblib"
            features_path = "/home/compomics/git/MLMarker/models/binary_features_TP_4000features_95to75missingness_2024.txt"           
        else:
            model_path = "/home/compomics/git/MLMarker/models/TP_full_92%_10exp_2024.joblib"
            features_path = "/home/compomics/git/MLMarker/models/features_TP_full_92%_10exp_2024.txt"
        if dev:
            model_path = "/home/compomics/git/MLMarker/models/NSAF_full_202412.joblib"
            features_path = '/home/compomics/git/MLMarker/models/NSAF_full_202412.txt'   
            self.dev = dev 
              
        self.model_path = model_path
        self.features_path = features_path
        self.model, self.features = self.load_model_and_features(model_path, features_path)
        self.sample = self.read_sample(sample_df)
        self.binary = binary        
        self.explainer = shap.TreeExplainer(self.model)  # Load SHAP explainer once
        self.zero_shaps = self.zero_sample()


    def load_model_and_features(self, model_path, features_path):
        self.model = joblib.load(self.model_path)
        if isinstance(self.model, xgb.XGBClassifier):    
            # Load model using XGBoost’s native load_model method
            self.model = xgb.XGBClassifier()
            self.model.load_model("/home/compomics/git/MLMarker/models/binary_TP_XGB_95to75missingness_2024.json")
            print('its an xgbclassifier')
        with open(features_path, 'r') as features_file:
            self.features = features_file.read().split(',\n')
            # self.features = self.features[:-1]
        return self.model, self.features
    
    def read_sample(self, sample_df):
        # sample should be one row only! otherwise give error
        if sample_df.shape[0] > 1:
            print("Error: Sample should be a single row for prediction, only first row is used")

        # match features between sample column names and self.features
        matched_features = list(set([feature for feature in self.features if feature in sample_df.columns]))
        added_features = list(set([feature for feature in self.features if feature not in sample_df.columns]))
        removed_features = list(set([feature for feature in sample_df.columns if feature not in self.features]))

        if len(added_features) > 0:
            print("Warning: {} model features are not in the sample and were added as zero values".format(len(added_features)))
        if len(removed_features) > 0:
            print("Warning: {} sample features are not in the model and were removed".format(len(removed_features)))

        # the final sample contains matched and added features
        sample_df = sample_df[matched_features]
        added_features_df = pd.DataFrame(0, index=sample_df.index, columns=added_features)
        sample = pd.concat([sample_df, added_features_df], axis=1)

        # remove columns in sample that are not in self.features
        sample = sample[self.features]

        if list(sample.columns) != self.features:
            print("Error: Sample columns do not match model features")
            print(f"Error: Features are {len(sample.columns)} and should be {len(self.features)}")
            print(len(sample.columns), len(set(sample.columns)))

        # Remove duplicate columns
        sample = sample.loc[:, ~sample.columns.duplicated()]

        return sample
    
    def zero_sample(self):
        zero_sample = pd.DataFrame(np.zeros((1, len(self.features))), columns=self.features)
        zero_shaps = self.shap_values_df(sample=zero_sample, n_preds=100)
        return zero_shaps


    def predict_top_tissues(self, n_preds=5):
        probabilities = self.model.predict_proba(self.sample).flatten()
        classes = self.model.classes_
        result = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)[:n_preds]
        formatted_result = [(pred_tissue, round(float(prob), 4)) for pred_tissue, prob in result]
        return formatted_result

    def calculate_shap(self, sample=None):
    
        """Calculate SHAP values for a given sample, or use the class sample by default."""
        if sample is None:
            sample = self.sample
        shap_values = self.explainer.shap_values(self.sample, check_additivity=False)
        #transpose so that the shape is always (1,35,4384)
        original_order = np.array(shap_values).shape
        classes = self.model.classes_
        desired_order = (original_order.index(1), original_order.index(len(classes)), original_order.index(len(self.features)))
        shap_values = np.transpose(shap_values, desired_order)
        shap_values = shap_values[0]  # remove the first dimension
        return shap_values

    
    def shap_values_df(self, sample=None, n_preds=5):
        """Get a dataframe with the SHAP values for each feature for the top n_preds tissues"""
        shap_values = self.calculate_shap(sample)
        classes = self.model.classes_
        predictions = self.predict_top_tissues(n_preds)
        
        shap_df = pd.DataFrame(shap_values)
        shap_df.columns = self.features
        shap_df['tissue'] = classes
        shap_df = shap_df.set_index('tissue')
        shap_df = shap_df.loc[[item[0] for item in predictions]]
        return shap_df
    
    def adjusted_shap_values_df(self, n_preds=5, penalty_factor=0.5):
        shap_df = self.shap_values_df(n_preds=n_preds)
        missing_proteins = self.sample.columns[self.sample.iloc[0] == 0]
        present_shap = shap_df.drop(missing_proteins, axis=1)
        absent_shap = shap_df[missing_proteins]
        notcontributing = absent_shap.loc[:, absent_shap.sum() == 0] 
        absent_shap_and_contributing = absent_shap.loc[:, absent_shap.sum() != 0]
        zero_absent_shap = self.zero_shaps[absent_shap_and_contributing.columns.tolist()]
        penalty = absent_shap_and_contributing - (penalty_factor *zero_absent_shap)
        combined_df = pd.concat([present_shap, notcontributing, penalty], axis=1)
        return combined_df
        
    def adjusted_absent_shap_values_df(self, n_preds=5, penalty_factor=0.5):
        """
        Adjust SHAP values by penalizing absent features based on a penalty factor.
        Keeps SHAP values for present features unchanged and handles contributing absent features separately.
        
        Args:
            n_preds (int): Number of top predicted tissues to include.
            penalty_factor (float): Factor to penalize SHAP values for absent features that contribute.

        Returns:
            pd.DataFrame: Adjusted SHAP values for the top predicted tissues.
        """
        # Get original SHAP values dataframe
        shap_df = self.shap_values_df(n_preds=n_preds)
        
        # Identify proteins that are absent (value == 0) in the sample
        absent_proteins = self.sample.columns[self.sample.iloc[0] == 0]
        present_proteins = [col for col in shap_df.columns if col not in absent_proteins]
        
        # Separate SHAP values for present and absent features
        present_shap = shap_df[present_proteins]  # SHAP values for present features remain unchanged
        absent_shap = shap_df[absent_proteins]
        
        # Handle absent features:
        # - Identify absent features that contribute (non-zero SHAP values)
        # - Penalize them using the penalty factor and pre-stored zero SHAP values
        contributing_absent_proteins = absent_shap.columns[absent_shap.sum() != 0]
        non_contributing_absent_proteins = absent_shap.columns[absent_shap.sum() == 0]
        
        # Penalize contributing absent features
        if len(contributing_absent_proteins) > 0:
            zero_absent_shap = self.zero_shaps[contributing_absent_proteins]  # Reference zero SHAP values
            penalized_absent_shap = absent_shap[contributing_absent_proteins] - (penalty_factor * zero_absent_shap)
        else:
            penalized_absent_shap = pd.DataFrame(columns=contributing_absent_proteins)  # Empty if no contributing absent features
        
        # Combine present SHAP values, penalized absent SHAPs, and non-contributing SHAPs
        combined_df = pd.concat(
            [
                present_shap,
                absent_shap[non_contributing_absent_proteins],  # Non-contributing SHAP values remain as is
                penalized_absent_shap,  # Adjusted SHAP values for contributing absent features
            ],
            axis=1
        )
        
        # Reorder to match original column order
        combined_df = combined_df[shap_df.columns]
        
        return combined_df

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

    def visualize_radar_chart(self):

        shap_df = self.adjusted_shap_values_df(n_preds=100, penalty_factor=0.5)
        predictions = shap_df.sum(axis=1).sort_values(ascending=False)
        prediction_df = pd.DataFrame(predictions)
        prediction_df.reset_index(inplace=True)
        prediction_df.columns = ['tissue', 'prob']
        # if prob negative, set to 0
        prediction_df.loc[prediction_df['prob'] < 0, 'prob'] = 0
        prediction_df['prob'] = prediction_df['prob'] *100
        prediction_df = prediction_df.sort_values(by='tissue')
        fig = px.line_polar(prediction_df, r='prob', theta='tissue', line_close=True)
        fig.show()

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
            sample_abundances = list(self.sample.values.flatten())
            plt.figure(figsize=(5,5))
            plt.scatter(sample_abundances, tissue_shap, c=['black' if abundance == 0 else 'blue' for abundance in sample_abundances])
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
    
    def confidence_score_df(self, n_preds=5):
        """Get a dataframe with the confidence score         
        Confidence Scoreclass = Probabilityclass × (∑(Positive Present Features)−∑(Negative Present Features)/Total Present Features)−(∑(Positive Absent Features)+∑(Negative Absent Features)/Total Features)
        """
        shap_values = self.calculate_shap()
        classes = self.model.classes_
        predictions = self.predict_top_tissues(n_preds)
        shap_df = self.shap_values_df(n_preds)

        # Get the SHAP values for the sample
         # Assuming shap_values_df aligns with the sample index
        shap_df = shap_df.T
        # Initialize a list to store results for this sample
        sample_results = []
        # Loop through each predicted tissue class and its probability
        for item in predictions:
            tissue, prob = item
            # For each tissue prediction, we calculate the contribution of SHAP values
            print(tissue, prob)
            # Separate SHAP values based on present/absent features
            positive_present = shap_df[(shap_df[tissue] > 0) & (sample_data.T != 0)].sum().values[0]
            negative_present = shap_df[(shap_df[tissue] < 0) & (sample_data.T != 0)].sum().values[0]
            
            positive_absent = shap_df[(shap_df[tissue] > 0) & (sample_data.T == 0)].sum().values[0]
            negative_absent = shap_df[(shap_df[tissue] < 0) & (sample_data.T == 0)].sum().values[0]
            
            # Total present and total features
            total_present = (sample_data.T != 0).sum().values[0]
            total_features = sample_data.T.shape[1]  # Total number of features
            
            # Calculate the confidence score using the formula
            confidence_score = prob * ((positive_present - negative_present) / total_present) - \
                                ((positive_absent + negative_absent) / total_features)
            
            # Append the result for this tissue to the list
            sample_results.append({
                'sample': sub_df.index[i],  # Sample name
                'tissue': tissue,
                'probability': prob,
                'confidence_score': confidence_score
            })
        
        # Convert the results for the current sample into a DataFrame and append to the final results dataframe
        sample_df = pd.DataFrame(sample_results)
        results_df = pd.concat([results_df, sample_df], ignore_index=True)

    # Show the final results
        return results_df

    def calculate_NSAF(self, df, lengths):
        """Calculate NSAF scores for proteins"""
        df['count'] = df['count'].astype(float)
        df['Length'] = df['Length'].astype(float)
        df['SAF'] = df['count'] / df['Length']
        total_SAF = df['SAF'].sum()
        df['NSAF'] = df['SAF'] / total_SAF
        return df
