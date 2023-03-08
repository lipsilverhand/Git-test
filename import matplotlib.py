import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.inspection import plot_decision_boundary

iris = datasets.load_iris()

print("Features: ", iris.feature_names)
print("Targets: ", iris.target_names)
print("Data examples:")
print(iris.data[:5])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(0, 10, 100)
y = np.sin(x)
z = np.cos(x)

ax.plot(x, y, z)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Line Plot')
ax.view_init(elev=20, azim=35) 

plt.show()

pca = PCA(n_components=3)
X = iris.data
y = iris.target
reduced_data = pca.fit_transform(X)

indices = np.random.permutation(len(X))
X_shuffled = X[indices]
y_shuffled = y[indices]

split_index = int(len(X_shuffled) * 0.8) 
X_train = X_shuffled[:split_index]
y_train = y_shuffled[:split_index]
X_test = X_shuffled[split_index:]
y_test = y_shuffled[split_index:]

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

plot_decision_boundary(svm, X_test, y_test, cmap='RdBu')

y_pred = svm.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

print("Accuracy: ", accuracy)
print("Confusion Matrix:\n", confusion_matrix)
