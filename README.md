# E-commerce Product Categorization Model

## Project Overview

The goal of this project is to develop a machine learning model to categorize e-commerce products based on their descriptions and numerical features such as retail price and discounted price. This is a multi-class classification problem where the target variable is the product category.

## Dataset Description

### Train Dataset: `train_product_data.csv`
The training dataset consists of 15,000 records and 15 features. Here is a detailed description of each column:

1. **uniq_id**: 
   - **Description**: A unique identifier for each product.
   - **Purpose**: Acts as the primary key to distinguish each product record uniquely.

2. **crawl_timestamp**: 
   - **Description**: The timestamp when the product data was last scraped or collected.
   - **Purpose**: Helps identify the data's recency and track changes over time.

3. **product_url**: 
   - **Description**: The URL linking directly to the product's page on the eCommerce platform.
   - **Purpose**: Allows direct access to the product's information and purchasing page.

4. **product_name**: 
   - **Description**: The name or title of the product as displayed on the eCommerce platform.
   - **Purpose**: Provides a searchable and readable identification of the product.

5. **product_category_tree**: 
   - **Description**: The hierarchical structure representing the product's category on the platform.
   - **Purpose**: Useful for categorization, analysis, and filtering of products.

6. **pid**: 
   - **Description**: A unique identifier specific to the eCommerce platform for each product.
   - **Purpose**: Used to reference products internally on the platform.

7. **retail_price**: 
   - **Description**: The original or retail price of the product before any discounts.
   - **Purpose**: Helps understand the product's standard market value.

8. **discounted_price**: 
   - **Description**: The price of the product after applying any discounts or offers.
   - **Purpose**: Reflects the final price a customer would pay.

9. **is_FK_Advantage_product**: 
   - **Description**: Indicator if the product is part of the Advantage program.
   - **Purpose**: Denotes if the product has additional benefits like faster delivery or special quality checks.

10. **description**: 
    - **Description**: Detailed information about the product, including features, specifications, and usage.
    - **Purpose**: Helps customers understand the product's value proposition and unique selling points.

11. **product_rating**: 
    - **Description**: The product's overall rating on the platform, based on customer reviews.
    - **Purpose**: Indicates customer satisfaction and product quality.

12. **overall_rating**: 
    - **Description**: The aggregate rating of the product across different platforms or periods.
    - **Purpose**: Offers a comprehensive view of the product's reception.

13. **brand**: 
    - **Description**: The name of the brand or manufacturer of the product.
    - **Purpose**: Assists in brand-based analysis and filtering.

14. **product_specifications**: 
    - **Description**: Detailed specifications of the product, often in JSON or structured format.
    - **Purpose**: Provides technical and functional details to aid customer decision-making.

### Test Dataset: `test_data.csv`
The test dataset consists of approximately 2,500 records and 14 features similar to those in the training data, excluding the target variable (`product_category_tree`).

### Test Results: `test_results.csv`
This dataset contains the target (`product_category_tree`) for the test data and can be used to evaluate the model's performance.

## Preprocessing

### Handling Missing Values
Missing values in the `description` column were filled with empty strings to ensure that the TF-IDF vectorizer can process the text data without issues.

```python
data['description'] = data['description'].fillna('')
test_data['description'] = test_data['description'].fillna('')
```

### Feature Extraction

#### Text Features
The `description` column was processed using a TF-IDF vectorizer to convert the text data into numerical features suitable for machine learning algorithms. The TF-IDF vectorizer was limited to a maximum of 10,000 features to manage complexity.

```python
tfidf = TfidfVectorizer(max_features=10000)
X_description = tfidf.fit_transform(data['description'])
```

#### Numerical Features
The numerical features (`retail_price` and `discounted_price`) were standardized using the `StandardScaler` to ensure they have a mean of 0 and a standard deviation of 1. This scaling helps in achieving better model performance.

```python
scaler = StandardScaler()
X_numerical = scaler.fit_transform(data[['retail_price', 'discounted_price']])
```

### Combining Features
The text and numerical features were combined using the `hstack` function from the `scipy.sparse` library. This resulted in a final feature set that includes both the processed text data and scaled numerical features.

```python
X_combined = hstack([X_numerical, X_description])
```

## Model Training

### Train-Test Split
The combined feature set was split into training and validation sets using an 80-20 split to evaluate the model's performance.

```python
X_train, X_val, y_train, y_val = train_test_split(X_combined, y, test_size=0.2, random_state=42)
```

### Model Selection and Hyperparameter Tuning
Several machine learning models were considered, but this documentation focuses on the LightGBM model due to its efficiency and performance with large datasets. Hyperparameter tuning was performed using `GridSearchCV` to find the best parameters for the model.

#### Parameter Grid for LightGBM
- `n_estimators`: Number of boosting rounds (50, 100, 200).
- `learning_rate`: Learning rate (0.01, 0.1, 0.2).
- `num_leaves`: Maximum tree leaves for base learners (31, 50, 70).

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'num_leaves': [31, 50, 70]
}
```

#### Grid Search
Grid search was used to identify the best hyperparameters based on cross-validated performance.

```python
grid_search = GridSearchCV(estimator=lgb.LGBMClassifier(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
```

### Model Evaluation
The best model from the grid search was evaluated on the validation set. The evaluation metrics included accuracy and the classification report, which provides precision, recall, and F1-score for each class.

```python
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred)}")
print(f"Classification Report:\n{classification_report(y_val, y_val_pred)}")
```

## Test Data Processing and Predictions

The test data underwent the same preprocessing steps as the training data (filling missing values, TF-IDF transformation, and scaling numerical features). Predictions were made using the best model, and the results were saved to a CSV file.

```python
X_test_description = tfidf.transform(test_data['description'])
X_test_numerical = scaler.transform(test_data[['retail_price', 'discounted_price']])
X_test_combined = hstack([X_test_numerical, X_test_description])
y_test_pred = best_model.predict(X_test_combined)
test_data['predictions'] = y_test_pred
test_data.to_csv('/content/drive/MyDrive/Colab Notebooks/Dataset/Data Science Hackathon/test_data_predictions.csv', index=False)
```

## Hyperparameter Tuning with LightGBM

GridSearchCV was used to tune the hyperparameters of the LightGBM model. Here are the best parameters and scores:

```python
grid_search = GridSearchCV(estimator=lgb.LGBMClassifier(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
```

## Confusion Matrix

The confusion matrix for the validation set was generated to evaluate the model's performance in more detail.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
```

## Lessons Learned

There were a lot of things tried through the course of this hackathon that didn't turn out as expected. Here are a few of the most important lessons learned:

- **Text Preprocessing**: First and foremost, I tried out some simple text preprocessing techniques like removing stopwords and lemmatization. That really didn't enhance the model's performance. TF-IDF vectorization, at its default parameters, did well.
**Model Selection**: Several models like Random Forest, Logistic Regression, and SVM were tried. LightGBM performed the best due to its high efficiency on large datasets and its robustness to overfitting.
- **Tuning of Hyperparameters**: GridSearchCV was necessary for choosing the optimal hyper-parameters.

We notice that tuning parameters like `num_leaves` and `learning_rate` made a huge difference in the model's performance.
