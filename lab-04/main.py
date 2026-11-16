import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

data = pd.read_csv(url, header=None, na_values="?")
data = data[[21, 25]].rename(columns={21: "horsepower", 25: "price"}).dropna()

train = data.drop_duplicates(subset="horsepower").sample(n=10, random_state=42)
test = data.drop(index=train.index)

X_train = train[["horsepower"]]
y_train = train["price"]
X_test = test[["horsepower"]]
y_test = test["price"]

model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=9),
    LinearRegression()
)

model.fit(X_train, y_train)

print("Завдання 1. Overfitting")
print(f"R² на тренувальних даних: {model.score(X_train, y_train):.4f}")
print(f"R² на тестових даних: {model.score(X_test, y_test):.4f}")

x_curve = np.linspace(data["horsepower"].min(), data["horsepower"].max(), 400)
y_curve = model.predict(pd.DataFrame({"horsepower": x_curve}))

plt.figure(figsize=(10, 6))
plt.scatter(X_train["horsepower"], y_train, label="Тренувальні дані")
plt.scatter(X_test["horsepower"], y_test, label="Тестові дані", alpha=0.5)
plt.plot(x_curve, y_curve, color="red", label="Поліноміальна регресія")
plt.xlabel("Потужність двигуна (horsepower)")
plt.ylabel("Ціна автомобіля")
plt.title("Перенавчання моделі (Overfitting)")
plt.ylim(0, max(data["price"]) * 1.2)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lab-04/overfitting.png", dpi=150)
plt.close()

from sklearn.dummy import DummyRegressor

underfit = DummyRegressor(strategy="mean")
underfit.fit(X_train, y_train)

print("\nЗавдання 2. Underfitting")
print(f"R² на тренувальних даних: {underfit.score(X_train, y_train):.4f}")
print(f"R² на тестових даних: {underfit.score(X_test, y_test):.4f}")

y_underfit = underfit.predict(pd.DataFrame({"horsepower": x_curve}))

plt.figure(figsize=(10, 6))
plt.scatter(X_train["horsepower"], y_train, label="Тренувальні дані")
plt.scatter(X_test["horsepower"], y_test, label="Тестові дані", alpha=0.5)
plt.plot(x_curve, y_underfit, color="red", label="Недонавчена модель")
plt.xlabel("Потужність двигуна (horsepower)")
plt.ylabel("Ціна автомобіля")
plt.title("Недонавчання моделі (Underfitting)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lab-04/underfitting.png", dpi=150)
plt.close()

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split

ridge_data = pd.read_csv(url, header=None, na_values="?")
ridge_data = ridge_data[[21, 13, 16, 24, 25]].dropna()

X = ridge_data[[21, 13, 16, 24]]
y = ridge_data[25]

X_train_ridge, X_test_ridge, y_train_ridge, y_test_ridge = train_test_split(
    X, y, test_size=0.2, random_state=42
)

ridge_model = make_pipeline(StandardScaler(), Ridge())

parameters = {
    "ridge__alpha": [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(ridge_model, parameters, cv=4)
grid.fit(X_train_ridge, y_train_ridge)

print("\nЗавдання 3. GridSearch для Ridge Regression")
print(f"Найкраще значення alpha: {grid.best_params_['ridge__alpha']}")
print(f"Найкращий середній R² (Cross-Validation): {grid.best_score_:.4f}")
print(f"R² на тестових даних: {grid.score(X_test_ridge, y_test_ridge):.4f}")
