# proj4

Collaborators: Mingyao Gu, Arsam Ijaz, Ronald Lam, and Daniel Yoon

Last updated: June 13, 2023

## 1.0 Introduction
Since data-related (analytics, science, engineering) jobs are in high demand, employees are hard to retain.
Why is the study of interest? 
- Understand factors that contribute to an employee in the data science field leaving current employment (forecasting)
- Install potential measurements that help companies retain employees (prevention)
This is a superviser machine learning problem, with binary classification as its outcome.

## 2.0 Analysis
The notebooks that document the analytical procedure can be found in the [notebooks](notebooks/) folder, where they are numbered to indicate different parts of the study in sequence. The pipeline components repeatedly used are made into functions and grouped in the [`project_pipeline`](notebooks/project_pipeline.py) module to make the notebooks easier to read and navigate.

The analytical procedure can be broken down into the following steps:
- Establish a baseline model that consists of workflow components further expanded later on in the study.
- Investigate different options of handling missing values.
- Evaluate different models including a linear model (logistic regression), a non-linear model (SVC with RBF kernel), and an ensemble model (random forest) through stratified k-fold cross validation. A deep neural network model is also developed. 
- Extract important features for visualization as well as model optimization.

### 2.1 Baseline Workflow
The baseline workflow is detailed in [`0_prelim`](notebooks/0_prelim.ipynb), which is comprised of the initial preprocessing of the dataset and training of a logistic regression model. The original data is stored and read from a PostgreSQL database (as per project requirement), the schema for which is stored in the [resources](notebooks/) folder alongside all the transformed data. 

Excluding the row identifier (`enrollee_id`) and the target (`target`), there are a total of 12 features in the dataset, most of which are categorical and missing values to varying degrees, as shown in the table below. 

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
| `company_type` | categorical | 6140 |
| `last_new_job` | categorical | 423 |
| `training_hours` | numerical | 0 |

In preparation for machine learning algorithms, each categorical feature is converted into a one-hot representation. For features with high cardinality such as `city`, one-hot encoding will result in a large number of input features due to a large number of possible categories; as such, the categories with instance counts below a threshold (in `city`'s case, set at 200) are binned together. The missing values, on the other hand, is more problematic. As an initial pass, considering that all features with null values are categorical, the missing values for each feature are imputed with its mode. As for the numerical features, they are standardized using `StandardScaler`.  

The target column, with its 0's and 1's, is what the model aims to predict. 1's represent employees leaving their current employment and 0's employees staying at their current employment. The target is imbalanced. As a result, in order to achieve better model performance, the relative porportions of 1's and 0's are kept when splitting the dataset into training and test sets (by setting `stratify=y` for `train_test_split`). In addition, `RandomOverSampler` is employed to ensure there are an equal number of 1's and 0's for the training set.

Once the preprocessing steps are completed, a basic linear regression model is trained on the training data. The model is able to achieve 0.74 accuracy and 0.74 recall, which establishes a good starting point. Due to the imbalanced target values in the test data, the area under the receiver operating characteristic curve (ROC AUC) is used subsequently in place of accuracy to measure the model's ability to distinguish between the positive and negative classes. The closer the value is to 1, the better it is. Recall is also of interest because the model's ability to predict individuals leaving their current employmnet is an important objective of the analysis. ROC AUC and recall scores are the primary metrics to evaluate models for the rest of the study.

### 2.2 Handling Missing Values
Before trying out various machine learning models, the impact of different ways of handling missing values on the performance of the baseline logistic regression classification is investigated, its details documented in [`1_compare_preprocessing`](1_compare_preprocessing.ipynb). The methods include dropping all null values of the original dataset as well as imputing null values using Datawig.

Dropping all null values results in a dataset with only 8955 rows and in turn a slightly lower AUC ROC score (from 0.74 to 0.73) and a much lower recall score for predicting individuals leaving their current employment (from 0.74 to 0.63). Since a higher recall score is of interest, this approach of handling missing values is not considered subsequently.

Because all features missing values are categorical, Datawig is chosen due to its support for imputation of categorical features. Its `SimpleImputer` (not to be confused with the similarly named function from Scikit-Learn) allows specifying the input columns containing useful data for imputation, the output column to impute values for, and the output path storing model data and metrics \[[1](#references)\]. To provide more input columns for the algorithm, the missing values of those features with fewer than 500 null instances are removed, which results in a dataset of 18014 rows. Different ways of forming the training data, which involve randomly performing an 80-20 split on the dataset or using all non-null rows, are documented in [`1_datawig_training_with_nans`](1_datawig_training_with_nans.ipynb) and [`1_datawig_training_without_nans`](1_datawig_training_without_nans.ipynb). The performance metrics obtained with either Datawig imputation show that the logistic regression model results in slightly lower ROC AUC (from 0.74 to 0.72) and lower recall (from 0.74 to 0.69) in comparison with those obtained with mode imputation. Considering that the latter is also computationally less intensive, the [data cleaned](resources/cleaned_mode.csv) using mode imputation is used for the rest of the analysis.

### 2.3 Model Selection
Because predicting whether or not an employee is leaving their current job is a supervised binary classification task, the models chosen for it include a logistic regression model (linear), a support vector classifier (SVC) with RBF kernel (non-linear), and a random forest classifer (ensemble). To compare these models, Scikit-Learn's k-fold cross-validation feature is utilized, its details documented in [`2_cross_validation_lr_svc_rf`](2_cross_validation_lr_svc_rf.ipynb). More specifically, the dataset is randomly split into 10 non-overlapping subsets or folds, and each model is trained 10 times, picking a different fold for evaluation and using the other 9 folds for training every time. In addition, the aforemtioned preprocessing considerations, such as preserving the percentage of sample for each target class, over-sampling the minority target class by picking samples at random with replacement, and standardizing numerical features, are built into the cross-validation process using `pipeline` from `imblearn`. The performance metrics for each model are summarized in the following table. Cross validation demonstrates that logistic regression and RBF SVC yield similar performance metrics, with the former producing slightly higher ROC AUC and recall. Random forest performs poorly by comparison, producing a considerably lower recall than the other two models. Therefore, it is not recommended for this classification task.

| Model | ROC AUC | Recall |
| --- | --- | --- |
| Logistic Regression | 0.78 | 0.73 |
| RBF SVC | 0.77 | 0.72 |
| Random Forest | 0.74 | 0.47 |

To see if even higher ROC AUC and recall can be achieved, a deep neural network model with hyperparameter tuning is implemented with the metric objective set to maximizing ROC AUC. Its details are documented in [`2_dnn_search`](2_dnn_search.ipynb). Based on a single stratified random sampling of a training set and a test set, the best model hyperparameters found include a first layer of 100 neurons with two additional hidden layers, each with 30 and 40 neurons, respectively. The activation functions are set to `relu` for the hidden layers and `sigmoid` for the output layer. The model results in a ROC AUC score of 0.80 and a recall score of 0.75 (without cross-validation), which is promising and warrants future investigation but is beyond the current project scope. 

### 2.4 Feature Importance
Since logistic regression achieves the highest ROC AUC and recall scores of the cross-validated models investigated while offering high interpretability regarding the relative importance of features, it is used to extract features that are considered important by the model in the classification outcome. The details are documented in [`3_feature_importance`](3_feature_importance.ipynb). The important features include city development index, city, relevant experience, education level, and company size. These features in relation to target outcome is explored in the next section.

Another utility that comes out of feature importance analysis is that using only these features for the cross-validation process allows us to evaluate the model performance with lower input cost. Although only 20 features (including all one-hot representations with high importance) are used as a result of feature selection in comparison with 59 used originally, the ROC AUC and recall scores achieved by each model are preserved and in the case of random forest signifcantly improved in terms of recall, as shown by the table below.

| Model | ROC AUC | Recall |
| --- | --- | --- |
| Logistic Regression | 0.78 | 0.73 |
| RBF SVC | 0.77 | 0.73 |
| Random Forest | 0.74 | 0.70 |

## 3.0 Visualization
To gain a deeper understanding of the factors that contribute to an employee in the data science field leaving their current employment, visualizations were created using the matplotlib library. The data used for plotting the bar charts is from the "cleaned_mode.csv" file.

### Education Level
One of the factors that stood out in the feature importance score was the education level of the employees. 
The bar chart revealed that employees with a "Graduate" education level were more likely to leave compared to employees with other education levels. This finding aligns with the predictions made by the model (based on the Feature Importance score) and suggests that higher education levels may contribute to higher employee retention rates.

![Alt text](https://github.com/skullfort/proj4/blob/main/visualizations/Basedon_Education.png)

### Relevant Experience
Another important factor identified by the model was the relevance of experience. 
The visualization indicated that employees with relevant experience were more likely to stay at their current employment compared to those without relevant experience. This observation also supports the model's prediction and suggests that having relevant experience is a contributing factor in employee retention.
![Alt text](https://github.com/skullfort/proj4/blob/main/visualizations/Basedon_Relevant_Exp.png)

### City Development Index
The visualization showed a clear trend that as the CDI increased, the likelihood of employees staying at their current employment also increased. This finding supports the model's prediction and suggests that employees in cities with higher development indexes may have better job opportunities for career growth, leading to higher retention rates.
![Alt text](https://github.com/skullfort/proj4/blob/main/visualizations/Basedon_CDI1.png)

### Company Size
The visualization revealed that employees in companies with a size of 50-99 had the highest leave rate, contradicting the model's prediction that companies with less than 10 employees would have the highest leave rate. This finding emphasizes the importance of visualizing the data and testing with multiple models to uncover insights that may not align with initial expectations.
![Alt text](https://github.com/skullfort/proj4/blob/main/visualizations/Basedon_CompanySize.png)

These visualizations helps provide a clear representation of the relationships between different factors and employee retention, validating the predictions made by the model and highlighting areas that organizations can focus on to improve employee retention strategies.

## 4.0 Conclusion

## References

1. https://datawig.readthedocs.io/en/latest/source/API.html

## Appendix

User can find in the [notebooks](notebooks/) folder the following addition files:
- model data and metrics saved during Datawig imputation
- initial implementation of [a DNN model](notebooks/2_dnn.ipynb) and its performance
- exploration of another classification model, [LightGBM](2_lightgbm.ipyng)
