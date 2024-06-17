# NBA-Logistic-Regression
This project aims to predict whether an NBA player is an All-Star based on their total points, assists, and rebounds. The data came from Basketball Reference, and exploratory data analysis and data cleaning were performed using Google Sheets. The cleaned data was then used to build a logistic regression model in Python.

1. Data Collection
   - Pulled data from Basketball Reference for the 2023-24 season initially.
   - Imported the data into Google Sheets for initial exploratory data analysis.

2. Exploratory Data Analysis (EDA) in Google Sheets
   - Analyzed the dataset to understand relationships between variables.

3. Data Cleaning
   - Cleaned the dataset by removing null values.
   - Used VLOOKUP to match player IDs and indicate All-Star status (Boolean, 1 Being Allstar, 0 Not an Allstar)

4. Model Training
   - Imported necessary libraries such as pandas, sklearn, seaborn, and matplotlib.
   - Loaded the cleaned dataset and selected relevant features (points, assists, rebounds) and the target variable (All-Star status).
   - Initially trained the logistic regression model using data from the 2023-24 season
  ```
  Model: Logistic Regression
     Accuracy: 0.9806
     Confusion Matrix:
     [[149   2]
      [  1   3]]
     Classification Report:
                   precision    recall  f1-score   support

                0       0.99      0.99      0.99       151
                1       0.60      0.75      0.67         4

         accuracy                           0.98       155
        macro avg       0.80      0.87      0.83       155
     weighted avg       0.98      0.98      0.98       155
```
   - Improved the model by adding additional data from the two prior seasons was included:
```
     Model: Logistic Regression
     Accuracy: 0.9819
     Confusion Matrix:
     [[473   2]
      [  7  16]]
     Classification Report:
                   precision    recall  f1-score   support

                0       0.99      1.00      0.99       475
                1       0.89      0.70      0.78        23

         accuracy                           0.98       498
        macro avg       0.94      0.85      0.89       498
     weighted avg       0.98      0.98      0.98       498
```
Seaborn Vizualization of Data

<img width="400" alt="image" src="https://github.com/trevorlovet1/NBA-Logistic-Regression/assets/112558354/4f089f76-c863-4681-8343-b0c140c2ece6">

5. Analyzing Model Performance
   -  Including data from the two prior seasons improved the precision of the model for the All-Star class
   - The recall for the All-Star class dipped. This is likely due to the model becoming more conservative in its predictions which reduced false positives but also missed some true positives 

7.  Prediction Function
   - Created a function to predict whether a player is an All-Star based on user input for points, rebounds, and assists.
   - Implemented a user prompt to gather input and make predictions.


Future Work Includes:
- Improve the model by adding additional data
- Testing other features in the dataset to see if it has an effect of the model
- Create a GUI that makes it easier for end users to utilize the predictor

