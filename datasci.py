import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

model = DecisionTreeRegressor(random_state=1)
csv_path = r'C:\Users\Hp\Downloads\dardanel.csv'
df = pd.read_csv(csv_path)
df = df.dropna(axis=0)
three = ['Şimdi', 'Yüksek', 'Düşük']
y = df.Şimdi
X = df[three]
model.fit(X, y)
predictions = model.predict(X)
print(X)
