from sklearn.datasets import fetch_california_housing
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
california_housing_data = fetch_california_housing(as_frame=True)
ch_data_frame = california_housing_data.frame
print(ch_data_frame)

# Visulizing the relation between varibales
sns.pairplot(x_vars=['MedInc', 'AveRooms'], y_vars=['MedHouseVal'], data=ch_data_frame, height=5)
plt.show()