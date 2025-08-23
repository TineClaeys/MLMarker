import shap
import pandas as pd
import numpy as np
import plotly.express as px

class Explainability:
    def __init__(self, model, features, sample=None, penalty_factor=0, explainer=None):
        self.model = model
        self.features = features
        self.sample = sample  # Store sample
        # Initialize the explainer (default is TreeExplainer if not passed)
        self.explainer = explainer or shap.TreeExplainer(model)  
        self.zero_shaps = self.zero_sample()
        self.penalty_factor = penalty_factor

        if sample is not None:
            self.update_sample(sample)  # Ensure SHAP values are computed only when sample exists
  
    def get_base_value(self):
        """Get the base value (expected value) of the model.
        
        Returns:
            float or numpy.ndarray: Base value(s) of the model.
            For binary classification, returns a single value.
            For multiclass, returns an array of values per class.
        """
        return self.explainer.expected_value

    def get_base_value_for_class(self, class_name):
        """Get the base value for a specific class.
        
        Args:
            class_name: Name of the class to get base value for
            
        Returns:
            float: Base value for the specified class
        """
        class_idx = list(self.model.classes_).index(class_name)
        base_values = self.explainer.expected_value
        return base_values[class_idx] if isinstance(base_values, (list, np.ndarray)) else base_values

    def zero_sample(self):
        zero_sample = pd.DataFrame(np.zeros((1, len(self.features))), columns=self.features)
        zero_shaps = self.shap_values_df(sample=zero_sample, n_preds=100)
        return zero_shaps


    def predict_top_tissues(self, sample=None, n_preds=5):
        if not isinstance(n_preds, int):
            raise ValueError(f"n_preds should be an integer, got {type(n_preds)}.")
        probabilities = self.model.predict_proba(sample).flatten()
        classes = self.model.classes_
        result = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)[:n_preds]
        return [(pred_tissue, round(prob, 4)) for pred_tissue, prob in result]

    def calculate_shap(self, sample=None):
        """Calculate SHAP values for a given sample, or use the class sample by default.
        
        Args:
            sample: Optional DataFrame containing sample data
            
        Returns:
            numpy.ndarray: Processed SHAP values with shape (n_classes, n_features)
        """
        if sample is None:
            sample = self.sample
        shap_values = self.explainer.shap_values(sample, check_additivity=False)
        original_order = np.array(shap_values).shape
        
        classes = self.model.classes_
        desired_order = (original_order.index(1), original_order.index(len(classes)), original_order.index(len(self.features)))
        shap_values = np.transpose(shap_values, desired_order)
        shap_values = shap_values[0]  # remove the first dimension
        #round to 4 decimal 
        shap_values = shap_values.round(5)
        return shap_values


    def shap_values_df(self, sample=None, n_preds=5):
        """Get a dataframe with the SHAP values for each feature for the top n_preds tissues"""
        if sample is None:
            sample = self.sample
        shap_values = self.calculate_shap(sample)
        classes = self.model.classes_
        predictions = self.predict_top_tissues(sample, n_preds)
        
        shap_df = pd.DataFrame(shap_values)
        shap_df.columns = self.features
        shap_df['tissue'] = classes
        shap_df = shap_df.set_index('tissue')
        shap_df = shap_df.loc[[item[0] for item in predictions]]
        return shap_df

    # def adjusted_absent_shap_values_df(self, n_preds=5):
    #     """
    #     Adjust SHAP values by penalizing absent features based on a penalty factor.
        
    #     Args:
    #         n_preds (int): Number of top predicted tissues to include.
            
    #     Returns:
    #         pd.DataFrame: Adjusted SHAP values for the top predicted tissues.
    #     """
    #     # Get original SHAP values dataframe
    #     penalty_factor = self.penalty_factor
    #     shap_df = self.shap_values_df(n_preds=n_preds)
        
    #     # Identify proteins that are absent (value == 0) in the sample

    #     #get column names where value is zero
    #     column_names = self.sample.columns[self.sample.iloc[0] == 0]

    #     absent_proteins = self.sample.columns[self.sample.iloc[0] == 0]
    #     present_proteins = [col for col in shap_df.columns if col not in absent_proteins]
    #     # Separate SHAP values for present and absent features
    #     present_shap = shap_df[present_proteins]  # SHAP values for present features remain unchanged
    #     absent_shap = shap_df[absent_proteins]
    #     # Handle absent features:
    #     # - Identify absent features that contribute (non-zero SHAP values)
    #     # - Penalize them using the penalty factor and pre-stored zero SHAP values
    #     contributing_absent_proteins = absent_shap.columns[absent_shap.sum() != 0]
    #     non_contributing_absent_proteins = absent_shap.columns[absent_shap.sum() == 0]

    #     # Penalize contributing absent features
    #     if len(contributing_absent_proteins) > 0:
    #         zero_absent_shap = self.zero_shaps[contributing_absent_proteins]  # Reference zero SHAP values
    #         penalized_absent_shap = absent_shap[contributing_absent_proteins] - (penalty_factor * zero_absent_shap)
    #     else:
    #         penalized_absent_shap = pd.DataFrame(columns=contributing_absent_proteins)  # Empty if no contributing absent features
        
    #     # Combine present SHAP values, penalized absent SHAPs, and non-contributing SHAPs
    #     combined_df = pd.concat(
    #         [
    #             present_shap,
    #             absent_shap[non_contributing_absent_proteins],  # Non-contributing SHAP values remain as is
    #             penalized_absent_shap,  # Adjusted SHAP values for contributing absent features
    #         ],
    #         axis=1
    #     )
        
    #     # Reorder to match original column order
    #     combined_df = combined_df[shap_df.columns]
        
    #     return combined_df

    def get_shap_values(self, sample=None, n_preds=5):
        """Get SHAP values with optional penalty for absent features.
        
        Args:
            sample: Optional DataFrame containing sample data
            n_preds: Number of top predicted tissues to include
            
        Returns:
            pd.DataFrame: SHAP values for top predicted tissues, adjusted if penalty_factor > 0
        """
        if sample is None:
            sample = self.sample
            
        # Calculate base SHAP values
        shap_values = self.explainer.shap_values(sample, check_additivity=False)
        original_order = np.array(shap_values).shape
        classes = self.model.classes_
        
        # Process array shape
        desired_order = (
            original_order.index(1),
            original_order.index(len(classes)), 
            original_order.index(len(self.features))
        )
        shap_values = np.transpose(shap_values, desired_order)[0]
        shap_values = np.round(shap_values, 5)
        
        # Create DataFrame
        predictions = self.predict_top_tissues(sample, n_preds)
        shap_df = pd.DataFrame(shap_values, columns=self.features)
        shap_df['tissue'] = classes
        shap_df = shap_df.set_index('tissue')
        shap_df = shap_df.loc[[item[0] for item in predictions]]
        
        # If no penalty, return unadjusted values
        if self.penalty_factor == 0:
            return shap_df
            
        # Apply penalty to absent features
        absent_proteins = sample.columns[sample.iloc[0] == 0]
        present_proteins = [col for col in shap_df.columns if col not in absent_proteins]
        
        # Split and process
        present_shap = shap_df[present_proteins]
        absent_shap = shap_df[absent_proteins]
        contributing = absent_shap.columns[absent_shap.sum() != 0]
        non_contributing = absent_shap.columns[absent_shap.sum() == 0]
        
        # Adjust contributing absent features
        if len(contributing) > 0:
            zero_shap = self.zero_shaps[contributing]
            penalized_shap = absent_shap[contributing] - (self.penalty_factor * zero_shap)
        else:
            penalized_shap = pd.DataFrame(columns=contributing)
        
        # Combine all parts
        adjusted_df = pd.concat([
            present_shap,
            absent_shap[non_contributing],
            penalized_shap
        ], axis=1)[shap_df.columns]
        
        return adjusted_df

    def visualize_shap_force_plot(self, shap_values, sample=None, tissue_idx=None, n_preds=5, tissue_name=None):
        """
        Visualizes SHAP force plots for top predicted tissues or for a specific tissue.
        """

        predictions = self.predict_top_tissues_shap(sample, n_preds=n_preds)
        i = 0
        shap.initjs()
        # If tissue_name is provided, check if it's valid
        if tissue_name:
            if tissue_name not in self.model.classes_:
                raise ValueError(f"Tissue '{tissue_name}' is not a valid class in the model.")
            tissue_loc = list(self.model.classes_).index(tissue_name)
            print(f"Visualizing force plot for tissue: {tissue_name}")
            # Visualize force plot for the specified tissue
            display(shap.force_plot(self.explainer.expected_value[1], np.round(shap_values[tissue_loc], 5), sample, matplotlib=True))
        else:
            # If no tissue_name is provided, visualize for top n predicted tissues
            print("No specific tissue provided, visualizing force plots for top predicted tissues:")
            for tissue, _ in predictions:
                tissue_loc = list(self.model.classes_).index(tissue)
                print(f"Tissue: {tissue}")

                i += 1
                # Display force plot for each top tissue
                display(shap.force_plot(self.explainer.expected_value[1], np.round(shap_values[tissue_loc], 5), sample, matplotlib=True))
                if i == n_preds:
                    break
        
    def calculate_NSAF(self, df, lengths):
        """Calculate NSAF scores for proteins"""
        df['count'] = df['count'].astype(float)
        df['Length'] = df['Length'].astype(float)
        df['SAF'] = df['count'] / df['Length']
        total_SAF = df['SAF'].sum()
        df['NSAF'] = df['SAF'] / total_SAF
        return df

    def predict_top_tissues_shap(self, sample=None, n_preds=5):
        """Predict top tissues using SHAP values + base values instead of predict_proba.
        
        Args:
            sample: Optional DataFrame containing sample data
            n_preds: Number of top predictions to return
            
        Returns:
            list: List of tuples (tissue_name, probability) sorted by probability
        """
        # Get SHAP values and sum per class
        shapvalues = self.get_shap_values(n_preds=len(self.model.classes_)).T.sum(axis=0)
        shapvalues_df = shapvalues.to_frame().rename(columns={0: 'Shap Value'})
        
        # Get base values for each class
        basevalues = self.get_base_value()
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

