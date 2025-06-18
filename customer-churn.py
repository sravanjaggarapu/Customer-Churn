import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, accuracy_score


#data manipulation
customer_churn = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(customer_churn.head())

customer_5 = customer_churn.iloc[:,4]
print(customer_5.head())

customer_15 = customer_churn.iloc[:,16]
print(customer_15)

senior_male_electronic = customer_churn[(customer_churn["gender"]=="Male") & (customer_churn["SeniorCitizen"] == 1) & (customer_churn["PaymentMethod"] == "Electronic check")]

customer_total_tenure = customer_churn[(customer_churn["tenure"] > 70) | (customer_churn["MonthlyCharges"] > 100)]
print(customer_total_tenure)

two_mail_yes = customer_churn[(customer_churn["Contract"] == "Two year") & (customer_churn["PaymentMethod"] == "Mailed check") & (customer_churn["Churn"]=="Yes")]
print(two_mail_yes)

customer_333 = customer_churn.sample(n=333)
print(customer_333)

customer_churn["Churn"].value_counts()

#data visualization

plt.bar(customer_churn["InternetService"].value_counts().keys().tolist(), customer_churn["InternetService"].value_counts().tolist(), color="green")
plt.xlabel("Types of InternetServices")
plt.ylabel("Count")
plt.title("Distribution of internetService")
plt.show()


plt.hist(customer_churn["tenure"],bins=30)
plt.title("Distribution of tenure")
plt.show()


plt.scatter(x=customer_churn["tenure"],y=customer_churn["MonthlyCharges"], color="brown")
plt.title("Tenure vs Monthly charges")
plt.show()

customer_churn.boxplot(by=["Contract"], column=["tenure"])
plt.show()

#LinearRegression


x = customer_churn[["tenure"]]
y = customer_churn[["MonthlyCharges"]]
print(y.head(), x.head())
x_train,x_test,y_train, y_test = train_test_split(x,y, test_size=0.3, random_state = 0)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape )

regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

print(y_pred[:5],y_test[:5])
print("\n\n\nLinear Regressor Score:")
print(np.sqrt(mean_squared_error(y_test,y_pred)))

#logistic Regression
x = customer_churn[["MonthlyCharges"]]
y = customer_churn[["Churn"]]

x_train,x_test,y_train, y_test = train_test_split(x,y, test_size=0.35, random_state = 0)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape )

log_model = LogisticRegression()
log_model.fit(x_train,y_train)
y_pred = log_model.predict(x_test)

print("\n\n\nLogistic Regressor Confusion matrix and Score")
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

#decision tree classifier
x = customer_churn[["tenure"]]
y = customer_churn[["Churn"]]

x_train,x_test,y_train, y_test = train_test_split(x,y, test_size=0.20, random_state = 0)
my_tree = DecisionTreeClassifier()

my_tree.fit(x_train,y_train)
y_pred = my_tree.predict(x_test)

print("\n\n\nDecision Tree Classifier Confusion matrix and Score")
print(confusion_matrix(y_pred,y_test))
print(accuracy_score(y_test,y_pred))

#Random Forest Classifier
rf = RandomForestClassifier()

rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)

print("\n\n\nRandom Forest Classifier Confusion matrix and Score")
print(confusion_matrix(y_pred,y_test))
print(accuracy_score(y_pred,y_test))