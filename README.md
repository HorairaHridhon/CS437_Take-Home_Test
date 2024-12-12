
# Food Category Classification using Machine Learning

This project focuses on classifying food items into various categories based on their nutritional features using different machine learning models. The dataset used is `foodstruct_nutritional_facts.csv`, and the project involves preprocessing, model training, hyperparameter tuning, and evaluation.

---

## Project Overview

The primary objective is to classify food categories such as 'Baked Products', 'Meat', 'Vegetables', 'Fruits', and more based on nutritional data. The project uses a pipeline approach to streamline preprocessing and model training.

### Key Steps:
1. **Data Cleaning**: Handling missing values and dropping less informative features.
2. **Feature Selection**: Dropping highly correlated features based on correlation analysis.
3. **Model Training**: Using multiple classifiers like Logistic Regression, KNN, Naive Bayes, Decision Trees, and Random Forest.
4. **Hyperparameter Tuning**: Using `RandomizedSearchCV` for optimizing model parameters.
5. **Evaluation**: Measuring performance with the Weighted F1 Score to handle class imbalance.

---

## Files

- **`cs437_take_home_test.ipynb`**: Jupyter Notebook with the complete implementation.
- **`foodstruct_nutritional_facts.csv`**: Nutritional dataset for various food items.

---

## Dataset Overview

The dataset contains 59 columns, including nutritional information such as:

- **Calories**
- **Carbs**
- **Fats**
- **Protein**
- **Sugar**
- **Vitamins and Minerals**

### Target Feature
- **CategoryName**: The category to which each food item belongs (e.g., 'Baked Products', 'Fruits', 'Meat').

### Sample Categories
- Baked Products
- Meat
- Sweets
- Vegetables
- Fruits
- Seafood

### Handling Missing Values
- Columns with excessive missing values (more than 1000) were dropped.
- For remaining features, missing values were filled with the mean or zero, depending on the extent of missing data.

---

## Steps to Run the Project

### 1. Install Dependencies

Ensure you have Python and the necessary libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Launch the Jupyter Notebook

```bash
jupyter notebook cs437_take_home_test.ipynb
```

### 3. Follow the Notebook

Execute each cell to perform the following:

1. **Load the Data**: Import the dataset and explore its structure.
2. **Data Preprocessing**: Handle missing values and perform feature selection.
3. **Model Training**: Train models using Logistic Regression, KNN, Decision Trees, Random Forest, and Naive Bayes.
4. **Hyperparameter Tuning**: Optimize models using `RandomizedSearchCV`.
5. **Evaluation**: Evaluate the best model using the Weighted F1 Score.

---

## Important Code Sections

### 1. Data Preprocessing

- **Dropping Columns with High Missing Values**:
  ```python
  df_new1 = df.drop(columns=['FoodName', 'Omega-6-Gamma-linoleicacid', 'Omega-3-Eicosatrienoicacid', 
                             'Omega-6-Dihomo-gamma-linoleicacid', 'Omega-6-Linoleicacid', 
                             'Omega-6-Arachidonicacid'])
  ```

- **Filling Missing Values**:
  ```python
  threshold = 0.5 * len(df_new1)
  for column, missing_count in missing_values1.items():
      if missing_count < threshold:
          df_new1[column] = df_new1[column].fillna(df_new1[column].mean())
      else:
          df_new1[column] = df_new1[column].fillna(0)
  ```

### 2. Model Training Pipeline

```python
pipeline = Pipeline(steps=[
    ('scale', MinMaxScaler()),
    ('classify', LogisticRegression(class_weight='balanced'))
])
```

### 3. Hyperparameter Tuning

```python
grid_search = GridSearchCV(pipe_grid_search, param_grid, scoring='f1_weighted',
                           cv=cv_stratified, n_jobs=-1)
grid_search.fit(X_train, y_train_encoded)
```

### 4. Evaluation Metrics

Weighted F1 Score is used due to class imbalance:

```python
f1_weighted = f1_score(y_test_encoded, y_pred, average='weighted')
print("Weighted F1 Score on test set:", f1_weighted)
```

---

## Reflections

### 1. Target Selection

The target column `CategoryName` was selected because food categories have distinct nutritional characteristics. For example:

- **Meat** and **Seafood** are high in protein.
- **Fruits** and **Vegetables** are low in calories.
- **Sweets** are high in sugar.

### 2. Model Performance

The model's performance varied due to class imbalance:

- **Majority Classes**: Performed well (e.g., 'Baked Products').
- **Minority Classes**: Performed poorly (e.g., 'Baby Foods').

### 3. Metric Selection

- **Weighted F1 Score** was chosen due to the imbalanced dataset.
- Accuracy was avoided as it can be misleading in imbalanced scenarios.

### 4. Addressing Imbalance

Techniques such as **SMOTE (Synthetic Minority Over-Sampling Technique)** or undersampling could be used to improve model performance on minority classes.

---

## License
This project is licensed under the MIT License.

## Acknowledgments
- Dataset Source: [Kaggle - Food Nutritional Facts](https://www.kaggle.com/datasets/beridzeg45/food-nutritional-facts)
