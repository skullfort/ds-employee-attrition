# proj4

## 1.0 Introduction
Since data-related (analytics, science, engineering) jobs are in high demand, employees are hard to retain.
Why is the study of interest? 
- Understand factors that contribute to an employee in the data science field leaving current employment (forecasting)
- Install potential measurements that help companies retain employees (prevention)

## 2.0 Analysis

### 2.1 Methodology
Consists of exploratory analysis and machine learning
- Establish a baseline that consists of workflow components that are further expanded later on in the study
- Investigate different options of handling missing values
- Evaluate different models including a linear model (logistic regression), a non-linear model (SVC with RBF kernel), and an ensemble model (random forest) through K-fold cross validation. A deep neural network model is also developed. 
- Extract important features for visualization as well as model tuning

### 2.2 Baseline Workflow
This baseline worklfow is detailed in [

### 2.3 Handling Missing Values
To establish a baseline for our study, we 
- Baseline (imputation with mode)
- Drop all NAN's
- Selectively drop some NAN's and impute with Datawig

### 2.4 Cross Validation
- Logistic Regression
- Deep Neural Network
- Random Forest (highlight important features)

### 2.5 Feature Importance and Selection

## Visualization

## Conclusion

## Appendix

The [original data](resource/aug_train.csv) and [preprocessed data](resource/hr_job_change.csv) can be found in the `resource` folder. The analyses, including prelimiary exploration of data, cleaning of data, and model selection, can be found in the `notebook` folder.
