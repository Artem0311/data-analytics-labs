import pandas as pd
import numpy as np

filename = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

headers = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
    "num-of-doors", "body-style", "drive-wheels", "engine-location",
    "wheel-base", "length", "width", "height", "curb-weight",
    "engine-type", "num-of-cylinders", "engine-size", "fuel-system",
    "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm",
    "city-mpg", "highway-mpg", "price"
]

df = pd.read_csv(filename, names=headers)

print("ПОЧАТКОВІ ДАНІ")
print(df.head(5))

df = df.replace("?", np.nan)

print("\nКІЛЬКІСТЬ ПРОПУЩЕНИХ ЗНАЧЕНЬ ДО ОЧИЩЕННЯ")
print(df.isna().sum())

for column in ["normalized-losses", "bore", "stroke"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")
    average = df[column].mean()
    df[column] = df[column].fillna(average)
    print(f"Середнє значення {column}: {average:.6f}")

print("\nЗАВДАННЯ 1. ЗАМІНА NaN У STROKE")
print("Пропущених значень у stroke:", df["stroke"].isna().sum())

df["height"] = df["height"] / df["height"].max()

print("\nЗАВДАННЯ 2. НОРМАЛІЗАЦІЯ HEIGHT")
print(df["height"].head(5))

dummy_variable = pd.get_dummies(df["aspiration"], dtype=int)
dummy_variable = dummy_variable.rename(columns={
    "std": "aspiration-std",
    "turbo": "aspiration-turbo"
})
df = pd.concat([df, dummy_variable], axis=1)
df = df.drop(columns="aspiration")

print("\nЗАВДАННЯ 3. ІНДИКАТОРНІ ЗМІННІ ASPIRATION")
print(df[["aspiration-std", "aspiration-turbo"]].head(5))

for column in ["horsepower", "peak-rpm", "price"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

numeric_columns = df.select_dtypes(include="number").columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
df["num-of-doors"] = df["num-of-doors"].fillna(df["num-of-doors"].mode()[0])

print("\nЗАВДАННЯ 4. ПЕРЕВІРКА ПРОПУЩЕНИХ ЗНАЧЕНЬ")
print(df.isna().sum())
print("Загальна кількість пропусків:", df.isna().sum().sum())
