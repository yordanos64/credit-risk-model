import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df = X.copy()

        # convert time
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

        # snapshot date
        if self.snapshot_date is None:
            self.snapshot_date = df['TransactionStartTime'].max()

        # -------------------------
        # RFM FEATURES
        # -------------------------
        rfm = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (self.snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Amount': 'sum',
            'Value': 'sum'
        }).reset_index()

        rfm.columns = [
            'CustomerId',
            'recency',
            'frequency',
            'monetary_amount',
            'monetary_value'
        ]

        # -------------------------
        # EXTRA FEATURES
        # -------------------------
        agg = df.groupby('CustomerId').agg({
            'Amount': ['mean', 'std', 'max']
        })

        agg.columns = ['amount_mean', 'amount_std', 'amount_max']
        agg = agg.reset_index()

        # merge
        customer_df = rfm.merge(agg, on='CustomerId', how='left')

        # fix missing std
        customer_df['amount_std'] = customer_df['amount_std'].fillna(0)

        return customer_df


    def build_preprocessor(self):

        numeric_features = [
            'recency',
            'frequency',
            'monetary_amount',
            'monetary_value',
            'amount_mean',
            'amount_std',
            'amount_max'
        ]

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ]
        )

        return preprocessor