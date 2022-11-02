import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.9851107314989933
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.01, max_depth=3, min_child_weight=19, n_estimators=100, n_jobs=1, subsample=0.05, verbosity=0)),
    StackingEstimator(estimator=SGDClassifier(alpha=0.01, eta0=0.1, fit_intercept=False, l1_ratio=0.0, learning_rate="invscaling", loss="modified_huber", penalty="elasticnet", power_t=10.0)),
    GradientBoostingClassifier(learning_rate=0.5, max_depth=9, max_features=0.8500000000000001, min_samples_leaf=16, min_samples_split=11, n_estimators=100, subsample=0.7500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)