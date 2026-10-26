from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import pandas as pd

iris = load_iris()
flowers = pd.DataFrame(iris.data, columns= iris.feature_names)
flowers["species"]=iris.target_names[iris.target]

x= flowers.drop(columns="species")
y= flowers["species"]

x_train, x_test, y_train, y_test= train_test_split(
    x,y, test_size=0.2, random_state=42, stratify=y
)
scaler= StandardScaler()
x_train_scaled= scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train_scaled, y_train)
predictions=model.predict(x_test_scaled)

labels= ["setosa", "versicolor", "virginica"]
print("accuracy:", accuracy_score(y_test, predictions))
print("order:", labels)

print(confusion_matrix(y_test, predictions, labels= labels))
print("f1:", f1_score(y_test, predictions, average="macro"))