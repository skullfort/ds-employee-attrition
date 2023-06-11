# proj4
Since data-related (analytics, science, engineering) jobs are in high demand, employees are hard to retain.
Why is the study of interest? 
- Understand factors that contribute to an employee in the data science field leaving current employment (forecasting)
- Install potential measurements that help companies retain employees (prevention)

## Methodology
Apply exploratory analysis and machine learning
- Establish a baseline that consists of workflow components that are further expanded later on in the study
- Investigate different options of handling missing values
- Evaluate different models including a linear model (logistic regression), a non-linear model (SVC with RBF kernel), an ensemble model (random forest), and a deep neural network through K-fold cross validation
- Extract important features for visualization as well as model tuning

## Preprocessing
To establish a baseline for our study, we 
- Baseline (imputation with mode)
- Drop all NAN's
- Selectively drop some NAN's and impute with Datawig

## Model Training
- Logistic Regression
- Deep Neural Network
- Random Forest (highlight important features)

## Appendix

The [original data](resource/aug_train.csv) and [preprocessed data](resource/hr_job_change.csv) can be found in the `resource` folder. The analyses, including prelimiary exploration of data, cleaning of data, and model selection, can be found in the `notebook` folder.
