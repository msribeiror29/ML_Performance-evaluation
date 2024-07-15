# ML_Performance-evaluation.

##Insurance Cost Prediction.

The goal is to receive an insurance dataset and attempt to predict the cost of these insurances. To do this, we will need to preprocess the data, test various machine learning models, and finally choose the most performant one.

We will use several libraries and techniques such as Pandas, NumPy, Seaborn, Scikit-Learn, and hyperparameter optimization to train, evaluate, and improve model performance.

Let's recap the main topics that will be covered:

    Data preprocessing.
    Categorical variables.
    Linear Regression.
    Machine Learning models.
    Model tuning.
    Performance evaluation.
    
#Data Preprocessing.
    
The first step is to import the dataset and perform an exploratory analysis. We need to understand what variables are available, data types, missing values, and data distribution.
Some questions to guide the exploratory analysis:

    How many observations and variables are in the dataset?
    Are there any missing values? In which variables and how many?
    Are the variables numerical or categorical?
    Are there outliers in the data?
    
With this initial analysis, we can identify data issues and start preprocessing. 
Common operations include:

    Handling missing values: removal or imputation.
    Normalizing numerical variables.
    Encoding categorical variables.
    Removing outliers.
    
The goal is to have clean, preprocessed data ready to be used in machine learning models. A good practice is to split the dataset into training, validation, and test sets to prevent data leakage.

#Categorical Variables.

Categorical variables need special attention during preprocessing. Since machine learning algorithms work with numbers, we need to transform these variables into numerical values.
The main techniques for encoding categorical variables are:

    Integer labels: Assign an integer to each category (1, 2, 3…).
    One-hot encoding: Create binary (0 or 1) variables for each category.
    Mean encoding: Replace categories with the target's mean value.
    Frequency encoding: Use the frequency of categories as the value.

We should test which method works best for our problem. Additionally, variables with many categories may need to be grouped or discarded to avoid overcomplicating the model.

#Linear Regression.

Linear regression is a great benchmark for regression problems. It has the advantage of being simple, easy to interpret, and quick to train.
We will fit a linear regression model using explanatory variables to predict insurance costs. 
Some best practices include:

    Checking for multicollinearity between variables.
    Performing variable selection to avoid overfitting.
    Analyzing coefficients to understand the importance of each variable.
    Checking residuals to validate model assumptions.
    
By evaluating performance metrics on the test set, we have a baseline to compare with other models.
Key points to analyze in linear regression:

    Mean Squared Error (MSE).
    Mean Absolute Error (MAE).
    R-squared coefficient.

#Machine Learning Models
With linear regression as a benchmark, we can move on to more complex machine learning models. 
Some interesting algorithms for this case are:

    Decision Trees.
    Random Forests.
    Neural Networks.
    XGBoost.
    
The goal is to test these different models, using cross-validation to evaluate performance. This way, we can understand which one performs best for our problem.
Some points that can improve results:

    Model parameters (tuning).
    Variable selection.
    Balanced training data.
In the end, we choose the model with the best performance on the validation set.

#Model Tuning.

Small adjustments in model parameters can significantly improve results. This is known as model tuning.
Common parameters to tune include:

    Decision Trees: depth, number of leaves.
    Random Forests: number of trees.
    Neural Networks: architecture, regularization.
    XGBoost: learning rate, regularization.
    
We will test various configurations and evaluate their impact on performance. We will find a balance between complexity and overfitting through tuning.
