from sklearn.datasets import fetch_california_housing
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
california_housing_data = fetch_california_housing(as_frame=True)
ch_data_frame = california_housing_data.frame
# print(ch_data_frame)

# Visulizing the relation between varibales
# sns.pairplot(x_vars=['MedInc', 'AveRooms'], y_vars=['MedHouseVal'], data=ch_data_frame, height=5)
# plt.show()

# Stardarize the values of dataframe except the values in "MedHouseVal" to make date easer to undrestand this means that the mean of data would be 0 and standard deviation would be 1
scalerObject = StandardScaler() # creating scaler object
scaled_california_housing_data = scalerObject.fit_transform(ch_data_frame.drop('MedHouseVal', axis =1))
# convert scaled data to data frame
scaled_ch_data_frame = pd.DataFrame(scaled_california_housing_data, columns= ['MedInc', 'HouseAge', 'AveRooms','AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Latitude'])
# add 'MedHouseVal' column to this data frame
medHouseVal_column = ch_data_frame['MedHouseVal']

# print(scaled_ch_data_frame.describe().round(3))

# Split data to train data (80%) and test data (20%)
x_train, x_test, y_train, y_test = train_test_split(scaled_ch_data_frame, medHouseVal_column, test_size=0.2, random_state=412)
# print(y_test) 

# Create a Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

# Evaluating model
y_lr_model_train_predict = lr_model.predict(x_train) # predict based on training data 80%
y_lr_model_test_predict = lr_model.predict(x_test) # predict based on testing data 20%

# Evaluate based on predicting values for training data
lr_model_train_mean_squared = mean_squared_error(y_train, y_lr_model_train_predict)
lr_model_train_r2 = r2_score(y_train, y_lr_model_train_predict)

# Evaluate based on predicting values for training data
lr_model_test_mean_squared = mean_squared_error(y_test, y_lr_model_test_predict)
lr_model_test_r2 = r2_score(y_test, y_lr_model_test_predict)

# creata a data frame to save result of evaluating models
models_eval_df = pd.DataFrame(['Linear Regression', lr_model_train_mean_squared, lr_model_train_r2, lr_model_test_mean_squared, lr_model_test_r2]).transpose()
models_eval_df.columns =['Model','Training MSE','Training R2','Testing MSE', 'Testing R2']
print(models_eval_df)