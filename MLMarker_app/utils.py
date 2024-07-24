def sample_predict_class(test_sample, model):
    # Predict the class probabilities for a test sample and return the top 5 predictions
    values = model.predict_proba(test_sample).tolist()[0]
    classes = model.classes_.tolist()
    d = dict(zip(classes, values))
    d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    return list(d.items())[:5]

def sample_predict_class_adjusted(test_sample, model, baseline_df):
    # Predict the adjusted class probabilities for a test sample and return the top 5 predictions
    baseline_values = baseline_df[model.classes_].values.tolist()[0]
    actual_probs = model.predict_proba(test_sample).tolist()[0]
    adjusted_probs = [actual - baseline for actual, baseline in zip(actual_probs, baseline_values)]
    d = dict(zip(model.classes_, adjusted_probs))
    d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    return list(d.items())[:5]
