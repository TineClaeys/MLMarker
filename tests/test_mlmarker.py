import pytest
import pandas as pd
import numpy as np
import mlmarker 
import shap

# Fixtures
@pytest.fixture
def test_data():
    """Load real test data"""
    df = pd.read_csv("/home/tine/git/MLMarker/tests/test_sample.csv")
    return df.iloc[0:1, 6:]

@pytest.fixture
def model():
    """Create model instance"""
    return mlmarker.MLMarker(binary=False, penalty_factor=1)

@pytest.fixture
def loaded_model(model, test_data):
    """Create model instance with loaded sample"""
    model.load_sample(test_data)
    return model

class TestMLMarker:
    def test_initialization(self, model):
        """Test model initialization"""
        assert isinstance(model, MLMarker)
        assert hasattr(model, 'model')
        assert hasattr(model, 'features')
        assert hasattr(model, 'explainability')
        assert model.penalty_factor == 1

    def test_load_sample(self, model, test_data):
        """Test sample loading"""
        added_features = model.load_sample(test_data, output_added_features=True)
        assert model.sample is not None
        assert isinstance(added_features, list)
        assert model.sample.shape[0] == 1
        assert all(f in model.features for f in model.sample.columns)

    def test_get_model_features(self, loaded_model):
        """Test getting model features"""
        features = loaded_model.get_model_features()
        assert isinstance(features, list)
        assert len(features) > 0
        assert all(isinstance(f, str) for f in features)

    def test_get_model_classes(self, loaded_model):
        """Test getting model classes"""
        classes = loaded_model.get_model_classes()
        assert isinstance(classes, np.ndarray)
        assert len(classes) > 0

    def test_predict_top_tissues_shap(self, loaded_model):
        """Test SHAP-based prediction"""
        predictions = loaded_model.predict_top_tissues_shap(n_preds=5)
        assert isinstance(predictions, list)
        assert len(predictions) == 5
        assert all(isinstance(p, tuple) and len(p) == 2 for p in predictions)
        assert all(isinstance(p[1], float) and 0 <= p[1] <= 1 for p in predictions)

    def test_shap_force_plot(self, loaded_model):
        """Test SHAP force plot generation"""
        # Test with default parameters
        loaded_model.shap_force_plot(n_preds=2)
        # Test with specific tissue
        loaded_model.shap_force_plot(tissue_name=loaded_model.get_model_classes()[0])

    def test_radar_chart(self, loaded_model):
        """Test radar chart generation"""
        loaded_model.radar_chart()

class TestExplainability:
    def test_get_base_value(self, loaded_model):
        """Test getting base value"""
        base_value = loaded_model.explainability.get_base_value()
        assert base_value is not None
        assert isinstance(base_value, (float, np.ndarray))

    def test_get_base_value_for_class(self, loaded_model):
        """Test getting base value for specific class"""
        class_name = loaded_model.get_model_classes()[0]
        base_value = loaded_model.explainability.get_base_value_for_class(class_name)
        assert isinstance(base_value, (float, np.float64))

    def test_zero_sample(self, loaded_model):
        """Test zero sample creation"""
        zero_shaps = loaded_model.explainability.zero_sample()
        assert isinstance(zero_shaps, pd.DataFrame)
        assert zero_shaps.shape[1] == len(loaded_model.features)

    def test_calculate_shap(self, loaded_model):
        """Test SHAP value calculation"""
        shap_values = loaded_model.explainability.calculate_shap()
        assert isinstance(shap_values, np.ndarray)
        assert shap_values.shape[1] == len(loaded_model.features)

    def test_get_shap_values(self, loaded_model):
        """Test getting SHAP values with different parameters"""
        # Test without penalty
        shap_df = loaded_model.explainability.get_shap_values(n_preds=2)
        assert isinstance(shap_df, pd.DataFrame)
        assert shap_df.shape[1] == len(loaded_model.features)

        # Test with penalty
        loaded_model.explainability.penalty_factor = 0.5
        shap_df_penalty = loaded_model.explainability.get_shap_values(n_preds=2)
        assert isinstance(shap_df_penalty, pd.DataFrame)
        assert not (shap_df == shap_df_penalty).all().all()

    @pytest.mark.parametrize("n_preds", [1, 3, 5])
    def test_predict_top_tissues_shap_explainer(self, loaded_model, n_preds):
        """Test SHAP-based prediction with different n_preds"""
        predictions = loaded_model.explainability.predict_top_tissues_shap(n_preds=n_preds)
        assert isinstance(predictions, list)
        assert len(predictions) == n_preds
        assert all(isinstance(p[1], float) and 0 <= p[1] <= 1 for p in predictions)

    def test_calculate_nsaf(self, loaded_model):
        """Test NSAF calculation"""
        df = pd.DataFrame({
            'count': [10, 20],
            'Length': [100, 200]
        })
        result = loaded_model.explainability.calculate_NSAF(df, None)
        assert 'NSAF' in result.columns
        assert result['NSAF'].sum() == pytest.approx(1.0)

    def test_error_handling(self, loaded_model):
        """Test error handling"""
        # Test invalid tissue name
        with pytest.raises(ValueError):
            loaded_model.explainability.get_base_value_for_class('invalid_tissue')
        
        # Test invalid n_preds
        with pytest.raises(ValueError):
            loaded_model.explainability.predict_top_tissues_shap(n_preds='invalid')

if __name__ == '__main__':
    pytest.main(['-v'])