import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
dataset = pd.read_csv('D:/Venkat/Study/IT/ML/Naive_Bayes/card_transdata.csv')
print(dataset.shape)
print(dataset.head(5))
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]
print(X.head(5))
print(y.head(5))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
model = GaussianNB()
model.fit(X_train,y_train)
pred_y = model.predict(X_test)
print(pred_y)
print("Accuracy score of the model is: {0}%".format(accuracy_score(pred_y,y_test)*100))
distance_from_home = float(input("Enter value for distance_from_home"))
distance_from_last_transaction = float(input("Enter value for distance_from_last_transaction"))
ratio_to_median_purchase_price = float(input("Enter value for ratio_to_median_purchase_price"))
repeat_retailer = int(input("Enter value for repeat_retailer"))
used_chip = int(input("Enter value for used_chip"))
used_pin_number = int(input("Enter value for used_pin_number"))
online_order = int(input("Enter value for online_order"))
input_test = [[distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order]]
print(input_test)
test_result = model.predict(sc.transform(input_test))
print(test_result)
if test_result == 1:
    print("Transaction may be Fraudulant")
elif test_result == 0:
    print("Transaction seems to be Genuine")


