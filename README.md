# Regression Model for Video Game Sale 

![Video Games Sales](https://pics.filmaffinity.com/video_games_the_movie-784311979-large.jpg)

## Objective
The objective of this notebook is to build a regression model to predict video game sales based on multiple features. During the data preparation and preprocessing stages, several challenges were faced, which are described below along with the solutions implemented.

## Dataset
The dataset used for this analysis can be found [here on Kaggle](https://www.kaggle.com/datasets/gregorut/videogamesales).

## Problems Encountered and Solutions

### 1. **Missing Values**
   - **Problem**: The dataset had missing values in the `Year` and `Publisher` columns.
   - **Solution**:
     - **Year Column**: The missing `Year` values were imputed using the median value of the `Year` for each `Platform`. This method ensures that missing data within each platform is filled with a reasonable estimate.
     - **Publisher Column**: The missing `Publisher` values were imputed using the mode (most frequent value) of the `Publisher` column within each `Genre`. This helped fill the missing data with the most likely publisher for each genre.

   ```python
   df['Year'] = df.groupby('Platform')['Year'].apply(lambda x: x.fillna(x.median()))
   df['Publisher'] = df.groupby('Genre')['Publisher'].transform(lambda x: x.fillna(x.mode().iloc[0]))
   ```

### 2. **Handling Categorical Data**
   - **Problem**: The `Platform` and `Genre` columns were categorical, making it impossible to use them directly in machine learning models.
   - **Solution**: These columns were encoded into numerical values using `LabelEncoder`. This allows the model to interpret the categorical data as numerical inputs.

   ```python
   from sklearn.preprocessing import LabelEncoder
   label_encoder = LabelEncoder()
   for col in ["Platform", "Genre"]:
       df[col] = label_encoder.fit_transform(df[col])
   ```

### 3. **Correlated Features**
   - **Problem**: Some features had high correlations with the target variable (`Global_Sales`), which could lead to multicollinearity issues.
   - **Solution**: A correlation matrix was generated to identify highly correlated features. Based on the results, the model was focused on the most significant features (`NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`), while columns like `Platform`, `Genre`, and `Year` were excluded from direct use due to their weak correlation with the target variable.

   ```python
   corr_ = df.corr()
   sns.heatmap(corr_, annot=True, linewidths=.2, cmap='YlOrRd')
   ```

### 4. **Dropping Unnecessary Columns**
   - **Problem**: Some columns, like `Rank` and `Name`, were irrelevant for the regression model and could create confusion or overfitting.
   - **Solution**: These columns were dropped to ensure that the model focused only on relevant features.

   ```python
   df.drop(columns=['Rank', 'Name'], inplace=True)
   ```
### 5. **Overfitting**
   - **Problem**: The linear regression model showed overfitting, with an unrealistically high R² score (0.99999) on the training data, indicating that the model was likely too complex and not generalizing well to new data.
   - **Solution**: Ridge regression was applied to regularize the model and prevent overfitting. The regularization parameter `alpha` was tuned to control the complexity of the model, leading to a more reasonable R² score.
   
   ```python
   ridge_model = Ridge(alpha=950)
   ridge_model.fit(X_train, y_train)
   ridge_pred = ridge_model.predict(X_test)
  ```
   

