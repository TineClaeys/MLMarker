import shap
import pandas as pd
import numpy as np
import plotly.express as px

class Explainability:
    def __init__(self, model, features, sample, explainer=None):
        self.model = model
        self.features = features
        self.sample = sample  # Store sample
        # Initialize the explainer (default is TreeExplainer if not passed)
        self.explainer = explainer or shap.TreeExplainer(model)  
        self.zero_shaps = self.zero_sample()

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
        original_order = np.array(shap_values).shape
        
        classes = self.model.classes_
        desired_order = (original_order.index(1), original_order.index(len(classes)), original_order.index(len(self.features)))
        shap_values = np.transpose(shap_values, desired_order)
        shap_values = shap_values[0]  # remove the first dimension
        return shap_values


    def shap_values_df(self, sample=None, n_preds=5):
        """Get a dataframe with the SHAP values for each feature for the top n_preds tissues"""
        if sample is None:
            sample = self.sample
        shap_values = self.calculate_shap(sample)
        classes = self.model.classes_
        predictions = self.predict_top_tissues(n_preds)
        
        shap_df = pd.DataFrame(shap_values)
        shap_df.columns = self.features
        shap_df['tissue'] = classes
        shap_df = shap_df.set_index('tissue')
        shap_df = shap_df.loc[[item[0] for item in predictions]]
        return shap_df

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

    def visualize_shap_force_plot(self, sample=None, n_preds=5, tissue_name=None):
        shap_values = self.calculate_shap(sample)
        predictions = self.predict_top_tissues(sample, n_preds)
        shap.initjs()
        if sample is None:
            sample = self.sample
        if tissue_name:
            # Check if tissue_name is in the model's classes
            if tissue_name not in self.model.classes_:
                raise ValueError(f"Tissue '{tissue_name}' is not a valid class in the model.")
            
            tissue_loc = list(self.model.classes_).index(tissue_name)
            logger.info(f"Visualizing force plot for tissue: {tissue_name}")
            print(f"Visualizing force plot for tissue: {tissue_name}")
            display(shap.force_plot(self.explainer.expected_value[1], shap_values[tissue_loc], sample, matplotlib=True))
        else:
            # If tissue_name is not provided, display for the top predicted tissues
            logger.info("Visualizing force plots for top predicted tissues:")
            print("Visualizing force plots for top predicted tissues:")
            for tissue, _ in predictions:
                tissue_loc = list(self.model.classes_).index(tissue)
                logger.info(f"Tissue: {tissue}")
                print(f"Tissue: {tissue}")
                display(shap.force_plot(self.explainer.expected_value[1], shap_values[tissue_loc], sample, matplotlib=True))


    def visualize_radar_chart(self, sample=None):
        if sample is None:
            sample = self.sample
        shap_df = self.adjusted_absent_shap_values_df(n_preds=100, penalty_factor=0.5)
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

    def calculate_NSAF(self, df, lengths):
        """Calculate NSAF scores for proteins"""
        df['count'] = df['count'].astype(float)
        df['Length'] = df['Length'].astype(float)
        df['SAF'] = df['count'] / df['Length']
        total_SAF = df['SAF'].sum()
        df['NSAF'] = df['SAF'] / total_SAF
        return df
