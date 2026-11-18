import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv(r"C:\Users\shiva\Downloads\archive (4)\temperatures.csv")
print(df.head())

X = df[["YEAR"]]

months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

for month in months:
    y = df[month]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nMonth:", month)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

     model.fit(X, y)
    future = pd.DataFrame({"YEAR": [2020]})
    print("2020 Temperature:", round(model.predict(future)[0], 2))

model = LinearRegression()
model.fit(X, df["MAR"])

plt.scatter(X, df["MAR"])
plt.plot(X, model.predict(X))
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.title("March Temperature Prediction")
plt.show()
