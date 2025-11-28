import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# 1. Get the data
df = pd.read_csv(r"C:\\Users\\oloke\\Downloads\\New folder (7)\\Titanic.csv")

# 2. Basic Cleanup (The manual way)
# Drop columns that have too many missing values or aren't useful for basic prediction
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Fill missing Age values with the average age
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Fill the 2 missing Embarked values with the most common one ('S')
df['Embarked'] = df['Embarked'].fillna('S')

# 3. Convert text to numbers
# Map Sex to numbers: male=0, female=1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert Embarked to dummy variables (C, Q, S becomes columns)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Select features (X) and target (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

# 4. Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling is important for SVM and Logistic Regression
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Initialize the algorithms
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC()
}

# 6. Train and print results
print("Model Accuracy Results:")
print("-----------------------")
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    score = accuracy_score(y_test, predictions)
    results[name] = score
    print(f"{name}: {score:.2f}")
    

plt.figure(figsize=(10, 5),title='Model Accuracy Comparison')
plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange', 'red', 'purple'])
plt.title('Model Accuracy Comparison on Titanic Dataset')
plt.ylabel('Accuracy')
plt.ylim(0.7, 0.9) # Zoom in on the relevant range
plt.show()