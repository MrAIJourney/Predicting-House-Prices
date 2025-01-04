from sklearn.datasets import fetch_california_housing
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
california_housing_data = fetch_california_housing(as_frame=True)
ch_data_frame = california_housing_data.frame
print(ch_data_frame)

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
scaled_ch_data_frame = pd.concat([scaled_ch_data_frame, medHouseVal_column], axis=1)
print(scaled_ch_data_frame.describe().round(3))