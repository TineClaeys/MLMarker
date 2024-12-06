import shap
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Optional


class Explainability:
    def __init__(self, model, features, explainer=None):
        self.model = model
        self.features = features
        self.explainer = explainer or shap.TreeExplainer(model)

    def calculate_shap(self, sample: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values for a given sample.
        """
        shap_values = self.explainer.shap_values(sample, check_additivity=False)
        return np.array(shap_values)

    def shap_values_df(
        self, shap_values: np.ndarray, sample: pd.DataFrame, n_preds: int = 5
    ) -> pd.DataFrame:
        """
        Return a DataFrame of SHAP values for the top n_preds classes.
        """
        classes = self.model.classes_
        top_predictions = sorted(
            zip(classes, self.model.predict_proba(sample).flatten()),
            key=lambda x: x[1],
            reverse=True,
        )[:n_preds]

        shap_df = pd.DataFrame(shap_values)
        shap_df.columns = self.features
        shap_df["class"] = classes
        shap_df = shap_df.set_index("class")
        shap_df = shap_df.loc[[item[0] for item in top_predictions]]

        return shap_df

    def adjust_absent_shap_values(
        self,
        shap_df: pd.DataFrame,
        sample: pd.DataFrame,
        zero_shap: pd.DataFrame,
        penalty_factor: float = 0.5,
    ) -> pd.DataFrame:
        """
        Adjust SHAP values for absent features using a penalty factor.
        """
        absent_features = sample.columns[sample.iloc[0] == 0]
        contributing_absent = absent_features[shap_df[absent_features].sum() != 0]
        penalized_shap = (
            shap_df[contributing_absent]
            - penalty_factor * zero_shap[contributing_absent]
        )

        combined_df = pd.concat(
            [shap_df.drop(absent_features, axis=1), penalized_shap], axis=1
        )
        return combined_df[shap_df.columns]

    def visualize_force_plot(
        self, shap_values: np.ndarray, sample: pd.DataFrame, class_index: int
    ):
        """
        Visualize a SHAP force plot for a specific class.
        """
        shap.initjs()
        shap.force_plot(
            self.explainer.expected_value[class_index],
            shap_values[class_index],
            sample,
            matplotlib=True,
        )

    def visualize_radar_chart(
        self, shap_df: pd.DataFrame, n_preds: int = 100, penalty_factor: float = 0.5
    ):
        """
        Visualize SHAP results in a radar chart.
        """
        predictions = shap_df.sum(axis=1).sort_values(ascending=False)
        prediction_df = pd.DataFrame(predictions)
        prediction_df.reset_index(inplace=True)
        prediction_df.columns = ["class", "score"]
        prediction_df["score"] = prediction_df["score"].clip(lower=0) * 100
        prediction_df = prediction_df.sort_values(by="class")

        fig = px.line_polar(prediction_df, r="score", theta="class", line_close=True)
        fig.show()
