#%% imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_validate,
    cross_val_predict,
    permutation_test_score,
)
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

TASK = "rest"
#%% load data
burst_table = pd.read_csv("duration_profiles.csv")
participant_table = pd.read_csv("ds007526/participants.tsv", sep="\t", na_values="n/a")
table = participant_table.pivot_table(
    index="group",
    columns="sex",
    values="age",
    aggfunc=["mean", "std", "count"],
    margins=True
)
table = (
    table
    .rename(columns={"mean": "Mean age (years)", "std":"Age standard deviation (years)", "count": "Participants (n)"}, level=0)
    .rename(columns={"F": "Female", "M": "Male"}, level=1)
    .rename(index={"HC": "Healthy controls", "PD": "Parkinson's disease"})
    .rename_axis(index="Group", columns=[None, "Sex"])
    .round(1)
)
table.to_html("demographics.html") # save demographic stats into a table
#%%

threshold_columns = [str(threshold) for threshold in range(20, 100, 5)]
threshold_values = np.array(range(20, 100, 5), dtype=float)

# keep the resting condition, and only electrodes where a beta peak was found
burst_table = burst_table[(burst_table["task"] == TASK) & (burst_table["peak_found"] == True)].copy()

#%% reduce each set of 16 data points to a polynomial with 3 degrees of freedom
durations = burst_table[threshold_columns].values.astype(float)
polynomial_coefficients = np.polyfit(threshold_values, durations.T, deg=2)
burst_table["curvature"] = polynomial_coefficients[0]
burst_table["slope"] = polynomial_coefficients[1]
burst_table["intercept"] = polynomial_coefficients[2]

#%% construct features, comparing just demographics vs durations & demographics
wide_features = burst_table.pivot_table(
    index="subject",
    columns="electrode",
    values=["intercept", "slope", "curvature"],
    aggfunc="mean",
)
wide_features
wide_features.columns = [f"{electrode}_{feature_name}" for feature_name, electrode in wide_features.columns]
feature_columns = list(wide_features.columns)
feature_columns
full_table = participant_table.merge(wide_features, left_on="participant_id", right_index=True, how="inner")
full_table = full_table.dropna(subset=["group", "age"]).reset_index(drop=True)
full_table.replace({"M": 0.0, "F": 1.0}, inplace=True)
labels = (full_table["group"] == "PD").astype(int).values
durations_and_demographics_feature_matrix = full_table[feature_columns + ["age", "sex"]].values
demographics_only_feature_matrix = full_table[["age", "sex"]].values

#%% construct two random forest models
inner_cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
outer_cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
test_params = { 
    'classifier__max_features': [5, 15, 25], # number of features given to a tree
}
fixed_params = {
    'max_depth': 10, # max number of branching points in each tree
    'n_estimators': 100, # number of trees
    'random_state': 42,
}
durations_and_demographics_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(**fixed_params)),
])
durations_and_demographics_model = GridSearchCV(
    estimator=durations_and_demographics_pipeline,
    param_grid=test_params,
    cv=inner_cross_validation,
    scoring="roc_auc"
)
demographics_only_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(**fixed_params)),
])
demographics_only_model = GridSearchCV(
    estimator=demographics_only_pipeline,
    param_grid=test_params,
    cv=inner_cross_validation,
    scoring="roc_auc"
)

#%% cross validation
durations_and_demographics_results = cross_validate(
    durations_and_demographics_model,
    durations_and_demographics_feature_matrix,
    labels,
    cv=outer_cross_validation,
    scoring="roc_auc",
    n_jobs=-1
)
demographics_only_results = cross_validate(
    demographics_only_model,
    demographics_only_feature_matrix,
    labels,
    cv=outer_cross_validation,
    scoring="roc_auc",
    n_jobs=-1
)

print("ROC-AUC score using durations and demographics:")
print(durations_and_demographics_results["test_roc_auc"].mean())
print("ROC-AUC score using demographics only:")
print(demographics_only_results["test_roc_auc"].mean())

#%% compute and plot confusion matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

prediction_cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
predicted_probabilities = cross_val_predict(
    durations_and_demographics_model, durations_and_demographics_feature_matrix, labels, cv=prediction_cross_validation,
    method="predict_proba", n_jobs=-1,
)[:, 1]
predicted_labels = (predicted_probabilities >= 0.5).astype(int)

cm = confusion_matrix(labels, predicted_labels)
disp = ConfusionMatrixDisplay(cm, display_labels=["HC", "PD"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion matrix")
plt.show()

#%% permutation testing - is the classifier better than chance with random shuffled labels
true_score, permuted_scores, permutation_p_value = permutation_test_score(
    durations_and_demographics_model, durations_and_demographics_feature_matrix, labels,
    # demographics_only_model, demographics_only_feature_matrix, labels,
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=3),
    n_permutations=100, n_jobs=-1, random_state=0, verbose=True
)
print(f"\nPermutation test: AUC = {true_score:.3f}, mean permuted AUC = {permuted_scores.mean():.3f}, "
      f"p = {permutation_p_value:.4f}")

#%% plotting permutation test
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(permuted_scores, bins=30, color="0.7", edgecolor="white", label="permuted labels")
ax.axvline(true_score, color="red", lw=2, label=f"actual AUC = {true_score:.3f}, p = {permutation_p_value:.3f}")
ax.set(xlabel="ROC AUC", ylabel="Number of permutations")
ax.legend()
plt.tight_layout()
plt.show()

# %% plotting feature importances
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
feature_names = feature_columns + ["age", "sex"]

rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=0, n_jobs=-1)
rf.fit(durations_and_demographics_feature_matrix, labels)

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

order = np.argsort(importances)           # ascending, so the biggest ends up at the top of barh
top = order[-15:]                         # show the 15 most important

fig, ax = plt.subplots(figsize=(6, 5))
ax.barh(np.array(feature_names)[top], importances[top], color="C0", ecolor="0.6")
ax.set(xlabel="Mean decrease in impurity", title="Random forest feature importance")
plt.tight_layout()
plt.show()
