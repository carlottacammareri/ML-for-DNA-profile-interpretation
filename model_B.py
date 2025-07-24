#Model B -- Predicting rework success using DNA profile features + LR related features

#Import Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
import joblib
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns

#Load data
df = pd.read_excel('Model_B_data.xlsx')
df = df.dropna(subset=['success'])
df['success'] = df['success'].astype(int)

#Handle categorical variables
categorical_cols = ['category', 'NoC_H1','#_of_unknowns']
encoder = OneHotEncoder(sparse_output= False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
encoded_df.index = df.index
df = pd.concat([df.drop(columns= categorical_cols), encoded_df], axis= 1)

#Select relevant features
features = [
     'dna_volume','Expected_peak_height', 'Peak_height_variance', 'mixture_proportion_poi','concentration', 'concentration_y', 'total_allele_count_subthreshold_automated', 'maximum_allele_count_y_marker',
 'num_type_3_loci', 'total_allele_count', 'num_type_1_loci', 'num_type_2_loci', 'num_locus_dropout_y_markers', 'num_alleles_reduced_locim_inference', 'num_locus_dropout_autosomal', 'maximum_allele_count', 'num_remaining_peaks_after_inference', 'mac_plus_mac_subthreshold_automated', 'amel_imbalance', 'degradation_slope_v2', 'total_allele_count_y_markers', 'num_inferred_alleles', 'num_peaks_below_st', 'tac_plus_tac_subthreshold_automated', 'min_peak_height',
] + list(encoded_df.columns)

#Select features and sucess labels in the DataFrame
df = df[features + ['success']]

#Separate input features and target variables
X = df.drop(columns=['success'], errors='ignore')
y = df['success']
X = X.apply(pd.to_numeric, errors= 'coerce')

#Split dataset train/test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)

#Initialize Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

#Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'roc_auc': make_scorer(roc_auc_score)
}
cv_results = cross_validate(model, X,y,cv=skf, scoring=scoring)

for metric in scoring.keys():
    scores = cv_results[f'test_{metric}']
    print(f'{metric.capitalize()}: {scores.mean():.4f}')

#Train model 
model.fit(X_train, y_train)

#Predictions on test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

#Try RFE to select most important features, trained the model with those features. 
# rfe = RFE(estimator=model, n_features_to_select=10, step=1)
# rfe.fit(X_train, y_train)
# selected_features = X_train.columns[rfe.support_]
# X_train_selected = X_train[selected_features]
# X_test_selected = X_test[selected_features]

# modelfinal = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
# modelfinal.fit(X_train_selected, y_train)

# y_pred = modelfinal.predict(X_test_selected)
# y_pred_proba = modelfinal.predict_proba(X_test_selected)[:,1]

#Save Model
joblib.dump(model, 'modelB.pkl')

#Compute performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'ROC-AUC:{roc_auc:.4f}')

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.title('confusion matrix')
plt.tight_layout()
plt.show()

#Feature importance
if len(model.feature_importances_) == len(X.columns):
    features_importance_df = pd.DataFrame({
       'Feature': X.columns,
       'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

#Aggregate categorical columns in feature importance
    def map_to_original_features(col):
       for cat_col in categorical_cols:
          if re.match(f'^{re.escape(cat_col)}(_.*)?$', col):
             return cat_col
       return col

    features_importance_df['Original_feature'] = features_importance_df['Feature'].map(map_to_original_features)
    aggregated_importance = features_importance_df.groupby('Original_feature')['Importance'].sum().reset_index()
    aggregated_importance = aggregated_importance.sort_values(by='Importance', ascending=False)
    total = aggregated_importance['Importance'].sum()
    aggregated_importance['Importance (%)'] = aggregated_importance['Importance'] / total * 100

#Plot Feature Importance
    plt.figure(figsize=(16,12))
    plt.barh(aggregated_importance['Original_feature'],aggregated_importance['Importance (%)'], color='skyblue')
    plt.xlabel('Feature Importance Score', fontsize = 14)
    plt.ylabel('Features', fontsize = 14)
    plt.title('Feature Importance in rework success prediction - Model A')
    plt.gca().invert_yaxis()
    plt.xticks(fontsize = 8, rotation=0)
    plt.yticks(fontsize = 14, rotation=0)
    plt.tight_layout()
    plot = 'successnoconcl.png'
    plt.show()
else:
  print('error')