# proj4

## 1.0 Introduction
Since data-related (analytics, science, engineering) jobs are in high demand, employees are hard to retain.
Why is the study of interest? 
- Understand factors that contribute to an employee in the data science field leaving current employment (forecasting)
- Install potential measurements that help companies retain employees (prevention)
This is a superviser machine learning problem, with binary classification as its outcome.

## 2.0 Analysis
The notebooks that document the analytical procedure can be found in the [notebooks](notebooks/) folder, where they are numbered to indicate different parts of the study in sequence. The functions repeatedly referenced are grouped in the `project_pipeline` module to make the notebooks easier to read and navigate.

Consists of exploratory analysis and machine learning

Methodology
- Establish a baseline that consists of workflow components that are further expanded later on in the study
- Investigate different options of handling missing values
- Evaluate different models including a linear model (logistic regression), a non-linear model (SVC with RBF kernel), and an ensemble model (random forest) through K-fold cross validation. A deep neural network model is also developed. 
- Extract important features for visualization as well as model tuning

### 2.1 Baseline Workflow
This baseline workflow is detailed in [`0_prelim`](notebooks/0_prelim.ipynb), which consists of the initial preprocessing of the dataset and training of a logistic regression model. There are a few outstanding characteristics associated with this dataset.

Excluding the row identifier (`enrollee_id`) and the target (`target`), which are a total of 13 features, most of which are categorical and missing values to varying degrees, as shown in the table below. 

| Feature | Type | Count of missing values (out of 19158) |
| --- | --- | --- |
| `city` | categorical | 0 |
| `city_development_index` | numerical | 0 |
| `gender`| categorical | 4508 |
| `relevant_experience` | categorical | 0 |
| `enrolled_university` | categorical | 386 |
| `education_level` | categorical | 460 |
| `major_discipline`| categorical | 2813 |
| `experience` | categorical | 65 |
| `company_size` | categorical | 5938 |
| ` company_type` | categorical | 6140 |
| `last_new_job` | categorical | 423 |
| `training_hours` | numerical | 0 |

In preparation for machine learning algorithms, each categorical feature is converted into a one-hot representation. For features with high cardinality such as `city`, one-hot encoding will result in a large number of input features due to a large number of possible categories; as such, the categories with instance counts below a threshold (in `city`'s case, set at 200) are binned together. The missing values, on the other hand, is more problematic. As an initial pass, the missing values of each feature are imputed with its mode. As for the numerical values, they are scaled using 

The target column, with its 0's and 1's, is what the model aims to predict. 1's represent employees leaving their current employment and 0's employees staying at their current employment. The target is imbalanced. As a result, in order to achieve better model performance, the relative porportions of 1's and 0's are maintained for splitting the dataset into training and testing sets (by setting `stratify=y` for `train_test_split`). In addition, `RandomOverSampler` is employed to ensure there are an equal number of 1's and 0's for the training data.

### 2.2 Handling Missing Values
To establish a baseline for our study, we 
- Baseline (imputation with mode)
- Drop all NAN's
- Selectively drop some NAN's and impute with Datawig

### 2.3 Cross Validation
- Logistic Regression
- Deep Neural Network
- Random Forest (highlight important features)

### 2.4 Feature Importance and Selection

## Visualization

## Conclusion

## Appendix

The [original data](resource/aug_train.csv) and [preprocessed data](resource/hr_job_change.csv) can be found in the `resource` folder. The analyses, including prelimiary exploration of data, cleaning of data, and model selection, can be found in the `notebook` folder.
