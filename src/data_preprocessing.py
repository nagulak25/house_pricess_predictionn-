import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(path):
    return pd.read_csv(path)

def get_features_target(df):
   X = df.drop('Price', axis=1)
   y = df['Price']
   return X, y

def create_preprocessor():
    numeric_features = ['Area', 'Bedrooms', 'Bathrooms', 'Age']
    categorical_features = ['Location', 'Property_Type']

    numeric_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    return preprocessor