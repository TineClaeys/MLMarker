from mlmarker.constants import (
    BINARY_MODEL_PATH,
    BINARY_FEATURES_PATH,
    UPDATED_MULTI_CLASS_MODEL_PATH,
    UPDATED_MULTI_CLASS_FEATURES_PATH,
)
from mlmarker.explainability import Explainability
import shap
import pandas as pd
import numpy as np
import plotly.express as px
from mlmarker.utils import (validate_sample, load_model_and_features,
    get_protein_info, 
    get_hpa_info,
    get_go_enrichment,
    visualise_custom_plot,
    visualise_custom_tissue_plot,
    prediction_df_2tissues_scatterplot
)
class MLMarker:
    def __init__(self, sample_df=None, binary=False, explainer = None, penalty_factor=0):
        self.model_path = BINARY_MODEL_PATH if binary else UPDATED_MULTI_CLASS_MODEL_PATH
        self.features_path = BINARY_FEATURES_PATH if binary else UPDATED_MULTI_CLASS_FEATURES_PATH
        self.model, self.features = load_model_and_features(
            self.model_path, self.features_path)
        self.penalty_factor = penalty_factor
        self.explainability = Explainability(self.model, self.features, None, self.penalty_factor, explainer=explainer)
       
    def load_sample(self, sample_df, output_added_features=False):
        """
        Loads and validates a sample for prediction and explainability.
        
        Args:
            sample_df (pd.DataFrame): The input sample.
            output_added_features (bool, optional): Whether to return the added features.

        Returns:
            If `output_added_features` is True, returns a tuple (validated_sample, added_features).
            Otherwise, updates the instance's sample attribute.
        """
        validated_sample = validate_sample(self.features, sample_df, output_added_features)
        
        if output_added_features:
            self.sample, added_features = validated_sample  # Unpack tuple
            self.explainability.sample = self.sample  # Update explainer
            return added_features
        else:
            self.sample = validated_sample
            self.explainability.sample = self.sample  # Update explainer

    def get_model_features(self):
        """
        Returns the features expected by the model.

        Returns:
        - list: Features used for predictions.
        """
        return self.features

    def get_model_classes(self):
        """
        Returns the classes predicted by the model.

        Returns:
        - list: Classes the model can predict.
        """
        return self.model.classes_

    # def predict_top_tissues(self, n_preds=5):
    #     probabilities = self.model.predict_proba(self.sample).flatten()
    #     classes = self.model.classes_
    #     result = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)[
    #         :n_preds
    #     ]
    #     return [(pred_tissue, round(prob, 4)) for pred_tissue, prob in result]
    
    def predict_top_tissues_shap(self, sample=None, n_preds=5):
        """Predict top tissues using SHAP values + base values instead of predict_proba.
        Immediatly integrates penalty factor correction if any
        Args:
            sample: Optional DataFrame containing sample data
            n_preds: Number of top predictions to return
        Returns:
            list: List of tuples (tissue_name, probability) sorted by probability
        """
        # Get SHAP values and sum per class
        shapvalues = self.explainability.get_shap_values(n_preds=len(self.model.classes_)).T.sum(axis=0)
        shapvalues_df = shapvalues.to_frame().rename(columns={0: 'Shap Value'})
        
        # Get base values for each class
        basevalues = self.explainability.get_base_value()
        classes = self.model.classes_
        
        # Create DataFrame with base values
        basevalues_df = pd.DataFrame(columns=['Tissue', 'Base Value'])
        for i, cls in enumerate(classes):
            basevalues_df = pd.concat([
                basevalues_df, 
                pd.DataFrame({'Tissue': [cls], 'Base Value': [basevalues[i]]})
            ], ignore_index=True)
        
        # Merge SHAP values with base values
        predictions_df = pd.merge(
            shapvalues_df, 
            basevalues_df, 
            left_index=True, 
            right_on='Tissue'
        )
        
        # Calculate final probabilities and sort
        predictions_df['Probability'] = predictions_df['Shap Value'] + predictions_df['Base Value']
        predictions_df = predictions_df.sort_values(by='Probability', ascending=False)
        
        # Return top n_preds in same format as original predict_top_tissues
        top_predictions = predictions_df.head(n_preds)
        return [(tissue, round(prob, 4)) for tissue, prob in 
                zip(top_predictions['Tissue'], top_predictions['Probability'])]

    def shap_force_plot(self, n_preds=5, tissue_name=None):
        shap_values = self.explainability.calculate_shap(self.sample)
        predictions = self.predict_top_tissues_shap(n_preds)
        tissue_name = tissue_name
        for tissue, _ in predictions:
            tissue_idx = list(self.model.classes_).index(tissue)
        self.explainability.visualize_shap_force_plot(shap_values, self.sample, tissue_idx, n_preds, tissue_name)
    
    def radar_chart(self, sample=None):
        predictions = self.predict_top_tissues_shap(n_preds=10)
        predictions_df = pd.DataFrame(predictions, columns=['Tissue', 'Probability'])
        predictions_df.loc[predictions_df['Probability'] < 0, 'Probability'] = 0
        predictions_df['Probability'] = predictions_df['Probability'] *100
        predictions_df = predictions_df.sort_values(by='Tissue')
        fig = px.line_polar(predictions_df, r='Probability', theta='Tissue', line_close=True)
        fig.show()
