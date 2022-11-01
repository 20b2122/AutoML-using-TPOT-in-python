import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -253497018.9304778
exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.8, learning_rate=0.001, loss="ls", max_depth=10, max_features=0.8, min_samples_leaf=4, min_samples_split=6, n_estimators=100, subsample=0.25)),
        StackingEstimator(estimator=RidgeCV())
    ),
    ElasticNetCV(l1_ratio=0.6000000000000001, tol=1e-05)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
