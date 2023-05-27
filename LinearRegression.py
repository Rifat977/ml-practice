import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

time_studied = np.array([23,67,45,23,45,7,88,45,23]).reshape(-1,1)
scores = np.array([87,67,23,45,78,57,67,23,67]).reshape(-1,1)

model = LinearRegression()
model.fit(time_studied, scores)

plt.scatter(time_studied, scores)
plt.plot(np.linspace(0,70,100).reshape(-1,1), model.predict(np.linspace(0, 70, 100).reshape(-1,1)), 'r')
plt.show()