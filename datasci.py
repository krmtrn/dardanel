import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

model = RandomForestRegressor(random_state=1)
csv_path = r'C:\Users\Hp\Desktop\ALSAMAPP\dardanel.csv'
df = pd.read_csv(csv_path)
df = df.dropna(axis=0)
three = ['Şimdi', 'Yüksek', 'Düşük']
y = df.Şimdi.str.replace(',', '').astype(float)
X = df[three].map(lambda x: float(x.replace(',', '')) if isinstance(x, str) else x)
train_x, val_x, train_y, val_y = train_test_split(X, y, random_state = 1)  
model.fit(train_x,train_y)

best_leap_size = None
min_mae = float('inf')

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y)
    if my_mae < min_mae:
        min_mae = my_mae
        best_leap_size = max_leaf_nodes

print("Best leap size:", best_leap_size)
print("Minimum MAE:", min_mae)

predicted = model.predict(val_x)
mae = mean_absolute_error(predicted, val_y)
print(mae)
print(predicted)

final_model = RandomForestRegressor(max_leaf_nodes=best_leap_size, random_state=1)
final_model.fit(X, y)
print("Model is trained.")
predicted = final_model.predict(val_x)
print(predicted)
