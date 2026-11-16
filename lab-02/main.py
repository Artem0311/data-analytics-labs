import pandas as pd
from scipy.stats import pearsonr, f_oneway

headers = ["symboling", "normalized-losses", "make", "fuel type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data", names=headers, na_values="?")

for column in ["bore", "stroke", "compression-ratio", "horsepower", "wheel-base", "price"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("Завдання 1. Кореляційна матриця")
print(df[["bore", "stroke", "compression-ratio", "horsepower"]].corr())

print("\nЗавдання 2. Кореляція stroke та price")
stroke_corr = df["stroke"].corr(df["price"])
print(stroke_corr)

print("\nЗавдання 3. Лінійна залежність stroke та price")
print(f"Кореляція слабка додатна (r = {stroke_corr:.3f}), тому вираженої лінійної залежності не очікується.")

print("\nЗавдання 4. Середня ціна за body-style")
print(df.groupby("body-style")["price"].mean())

print("\nЗавдання 5. Кореляція Пірсона: wheel-base та price")
data = df[["wheel-base", "price"]].dropna()
result = pearsonr(data["wheel-base"], data["price"])
print("Pearson:", result.statistic, "P-value:", result.pvalue)

print("\nЗавдання 6. Кореляція Пірсона: horsepower та price")
data = df[["horsepower", "price"]].dropna()
result = pearsonr(data["horsepower"], data["price"])
print("Pearson:", result.statistic, "P-value:", result.pvalue)

print("\nЗавдання 7. Дисперсійний аналіз ANOVA")
for column in ["body-style", "drive-wheels", "aspiration"]:
    groups = [
        group["price"].dropna().to_numpy()
        for _, group in df.groupby(column)
        if group["price"].notna().sum() >= 2
    ]
    result = f_oneway(*groups)
    print(column, "F-test:", result.statistic, "P-value:", result.pvalue)
    print("Висновок:", "Виявлено статистично значущі відмінності середніх цін." if result.pvalue < 0.05 else "Статистично значущих відмінностей середніх цін не виявлено.")
