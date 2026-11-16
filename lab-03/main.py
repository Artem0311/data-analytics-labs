import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

df = pd.read_csv(url, header=None, na_values="?")
df = df.iloc[:, [1, 16, 24, 25]].copy()
df.columns = ["normalized-losses", "engine-size", "highway-mpg", "price"]
df = df.dropna()

y = df["price"]
results = {}

print("Кількість автомобілів для моделювання:", len(df))

print("\nЗавдання 1. Проста лінійна регресія")
X1 = df[["engine-size"]]
model1 = LinearRegression()
model1.fit(X1, y)
pred1 = model1.predict(X1)
results["Проста лінійна регресія"] = (
    r2_score(y, pred1),
    mean_squared_error(y, pred1)
)

print("\nЗавдання 2. Множинна лінійна регресія")
X2 = df[["normalized-losses", "highway-mpg"]]
model2 = LinearRegression()
model2.fit(X2, y)
pred2 = model2.predict(X2)
results["Множинна лінійна регресія"] = (
    r2_score(y, pred2),
    mean_squared_error(y, pred2)
)

print("\nЗавдання 3. Поліноміальна регресія 11-го степеня")
X3 = df[["highway-mpg"]]
model3 = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=11),
    LinearRegression()
)
model3.fit(X3, y)
pred3 = model3.predict(X3)
results["Поліноміальна регресія"] = (
    r2_score(y, pred3),
    mean_squared_error(y, pred3)
)

print("\nЗавдання 4. Порівняння моделей")
for name, (r2, mse) in results.items():
    print(f"{name}: R² = {r2:.4f}, MSE = {mse:.2f}")

best = min(results, key=lambda name: results[name][1])
print("\nНайкраща модель за MSE:", best)
