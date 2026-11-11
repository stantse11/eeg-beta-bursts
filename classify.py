#%%
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

TASK = "walk"
#%% load data
burst_table = pd.read_csv("duration_profiles.csv")
participant_table = pd.read_csv("ds007526/participants.tsv", sep="\t", na_values="n/a")

threshold_columns = [str(threshold) for threshold in range(20, 100, 5)]
threshold_values = np.array(range(20, 100, 5), dtype=float)

# Keep the resting condition, and only electrodes where a beta peak was found
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
wide_features.columns = [f"{electrode}_{feature_name}" for feature_name, electrode in wide_features.columns]
feature_columns = list(wide_features.columns)
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

# #%%
# # ----------------------------------------------------------------------
# # 6. Out-of-fold predictions, bootstrap confidence interval, confusion matrix, ROC curve
# # ----------------------------------------------------------------------
# prediction_cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
# predicted_probabilities = cross_val_predict(
#     burst_model, durations_and_demographics_feature_matrix, labels, cv=prediction_cross_validation,
#     method="predict_proba", n_jobs=-1,
# )[:, 1]
# predicted_labels = (predicted_probabilities >= 0.5).astype(int)

# observed_auc = roc_auc_score(labels, predicted_probabilities)

# random_generator = np.random.default_rng(0)
# bootstrap_aucs = []
# for bootstrap_index in range(2000):
#     resampled_indices = random_generator.integers(0, len(labels), len(labels))
#     if len(np.unique(labels[resampled_indices])) < 2:
#         continue
#     bootstrap_aucs.append(roc_auc_score(labels[resampled_indices], predicted_probabilities[resampled_indices]))
# lower_bound, upper_bound = np.percentile(bootstrap_aucs, [2.5, 97.5])

# print(f"\nOut-of-fold AUC: {observed_auc:.3f}  (95% bootstrap confidence interval {lower_bound:.3f} to {upper_bound:.3f})")
# print("Confusion matrix (rows = true healthy control / Parkinson's, columns = predicted):")
# print(confusion_matrix(labels, predicted_labels))

# false_positive_rate, true_positive_rate, _ = roc_curve(labels, predicted_probabilities)
# plt.figure(figsize=(5, 5))
# plt.plot(false_positive_rate, true_positive_rate, label=f"AUC = {observed_auc:.2f}")
# plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
# plt.xlabel("False positive rate")
# plt.ylabel("True positive rate")
# plt.title("Healthy control vs Parkinson's disease (out-of-fold)")
# plt.legend()
# plt.tight_layout()
# plt.savefig("roc_curve.png", dpi=200)

# #%%
# # ----------------------------------------------------------------------
# # 7. Permutation test: is the AUC better than chance?
# #    (shuffles labels 500 times and repeats the whole nested procedure each time;
# #     lower n_permutations if it runs too slowly)
# # ----------------------------------------------------------------------
# true_score, permuted_scores, permutation_p_value = permutation_test_score(
#     burst_model, durations_and_demographics_feature_matrix, labels,
#     scoring="roc_auc",
#     cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=3),
#     n_permutations=500, n_jobs=-1, random_state=0,
# )
# print(f"\nPermutation test: AUC = {true_score:.3f}, mean permuted AUC = {permuted_scores.mean():.3f}, "
#       f"p = {permutation_p_value:.4f}")

# #%%
# # ----------------------------------------------------------------------
# # 8. Which features drive the classifier? (fit on all data, for interpretation only)
# # ----------------------------------------------------------------------
# burst_model.fit(durations_and_demographics_feature_matrix, labels)
# final_coefficients = burst_model.best_estimator_.named_steps["classifier"].coef_[0]

# coefficient_table = pd.DataFrame({"feature": feature_columns, "coefficient": final_coefficients})
# coefficient_table["absolute_coefficient"] = coefficient_table["coefficient"].abs()
# coefficient_table = coefficient_table.sort_values("absolute_coefficient", ascending=False)

# print("\nBest regularization strength:", burst_model.best_params_["classifier__C"])
# print("Top 15 features by absolute standardized coefficient:")
# print(coefficient_table.head(15).to_string(index=False))
