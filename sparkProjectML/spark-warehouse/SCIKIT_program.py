import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
# Training data
input_data = np.array([[
             3, -1.5, 3, -6.4],
             [0, 3, -1.3, 4.1],
             [1, 2.3, -2.9, -4.3]])

data_standardized = preprocessing.scale(input_data)
print ("\nMean =", data_standardized.mean(axis=0))
print ("Std deviation =", data_standardized.std(axis=0))

data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(input_data)
print ("\nMin max scaled data =", data_scaled)

data_normalized = preprocessing.normalize(input_data, norm='l1')
print ("\nL1 normalized data =", data_normalized)

data_binarized = preprocessing.Binarizer(threshold=1.4).transform(input_data)
print("\nBinarized data =", data_binarized)



Diameter = [[6], [8], [10], [14], [18]]
price = [[7], [9], [13], [17.5], [18]]
print(np.var(Diameter, ddof=1))
#np.cov([Diameter], [price])


# Create and fit the model
model = LinearRegression()
model.fit(Diameter, price)
size_of_pizza = [[12]]
print("\n the pizza with size = ", size_of_pizza[0][0], "has predicted price of ", model.predict(size_of_pizza)[0][0])

exit(1)
plt.figure()
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(Diameter, price, 'k.')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()
