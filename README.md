Project Report – Parkinson’s Disease Detection Model

Project Description
This project focuses on detecting Parkinson’s disease in patients using machine learning techniques. Parkinson's disease is a progressive neurological disorder, and early detection can aid in managing and potentially slowing its progression. The code uses data preprocessing, feature scaling, and a machine learning model to classify patients as either having Parkinson’s or not based on specific features.
Machine Learning Model in This Project
The machine learning model used in this project is Support Vector Machine (SVM). SVM is a supervised learning algorithm used for classification and regression tasks. It works by finding a hyperplane that best divides the data into different classes. For our case, the two classes are individuals who have Parkinson’s Disease and those who do not.
Why SVM?
SVM is ideal for this classification task because:
1.	Linear separability: In cases where the data is linearly separable (i.e., the data can be divided with a straight line or hyperplane), SVM performs very well.
2.	Outlier handling: SVM is effective in handling outliers, which are common in medical datasets.
3.	High-dimensional space: SVM can work efficiently even with high-dimensional data (lots of features) which makes it suitable for this task, as the Parkinson’s dataset contains multiple features extracted from voice recordings.
Other classification models like Decision Trees or Random Forests were not used because they may be more prone to overfitting and may not generalize as well as SVM, especially when the data is not very large or when the features are highly correlated.
GitHub Link
GitHub Repository Link - ashwin-gs/AI_Parkinson-s_Detection_Model
Project Output Image
The project outputs a classification accuracy score, showing how accurately the model can identify Parkinson's patients from the given data. Additionally, the model produces a list of predictions indicating the likelihood that everyone in the test set has Parkinson’s disease.
  
Code Explanation with Step-by-Step Analysis
1. Importing Libraries
 
Explanation:
Loading Data: The data is loaded into a Pandas DataFrame from a CSV file named parkinsons.csv using the pd.read_csv() method.
Exploring the Data: We first explore the data by looking at the first few rows with head(), the shape of the data with shape, and checking for any missing values.

2. Loading and Inspecting the Data
 
The dataset is loaded into a Pandas DataFrame. To understand the data structure, we view the first few rows and check the dataset’s shape, which reveals the number of features and samples available for analysis. This inspection is important for verifying that the data has loaded correctly and identifying any potential data-cleaning needs.
3. Splitting Data into Features and Target
 
The dataset is divided into two parts: X (features) and y (target variable). Here, status is our target variable, representing whether an individual has Parkinson’s disease (1 for positive, 0 for negative).
4. Data Standardization
 
Explanation: We use the StandardScaler to standardize the features. Standardization ensures that the data is centered around 0, which helps the SVM algorithm perform better.
5. Train-Test Split
 
The data is split into training and test sets in an 80-20 ratio. This separation allows us to assess how well the model generalizes to new, unseen data. A random_state is set for reproducibility.
6. Training the SVM Classification Model
 
The SVC() function from Scikit-Learn is used to create an SVM model with a linear kernel. This model is then trained using the training data (X_train, Y_train).
7. Evaluating Model Performance 
After training, the model makes predictions on the test set. The accuracy_score function calculates the accuracy, providing a simple metric to evaluate how often the model’s predictions are correct. This score helps assess the model’s overall performance and potential areas for improvement.
8. Making Predictions
 
Once the model is trained and evaluated, we can use it to make predictions on new, unseen data. In this example, input data representing a person's medical features is provided, which the model uses to predict whether the person has Parkinson’s Disease or not.

Model Training and Output
The model achieved an accuracy score of approximately 88.46% on the test data. This result demonstrates that SVM classification effectively identifies Parkinson’s cases in this dataset, showcasing its suitability for this binary classification problem.

Miscellaneous Notes
Why Not Other Models? 
We chose the SVM model for its efficiency and ability to work well with both linear and non-linear data. Decision Trees and Random Forests, while also good models, tend to overfit on smaller datasets. Also, SVM tends to be more effective in high-dimensional spaces, which is often the case with medical datasets.
Impact of SVM on Accuracy: 
By using SVM, we achieved a relatively high accuracy score on both the training and test data. This shows that the SVM algorithm is appropriate for this problem and has a good ability to generalize. Other models like Logistic Regression or K-Nearest Neighbours might have had lower accuracy due to overfitting or underfitting.

Conclusion
This project successfully demonstrates how machine learning, specifically using the Support Vector Machine (SVM) algorithm, can be employed to detect Parkinson's Disease based on medical features. The model performs well on both the training and testing data, providing reliable predictions. This approach can be extended to real-world applications, potentially helping in early detection and diagnosis of Parkinson’s Disease using machine learning models.


