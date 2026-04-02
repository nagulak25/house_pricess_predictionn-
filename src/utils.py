def get_feature_importance(model):
    try:
        importance = model.named_steps['model'].feature_importances_
        return importance
    except:
        return "Feature importance not available"