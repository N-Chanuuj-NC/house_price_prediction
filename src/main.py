import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
# df = pd.read_csv('data/Housing.csv')
df = pd.read_csv('../data/Housing.csv')

# Describe--> To find the statistical description
df.describe()

#To Find the Null and datatype details
df.info()

#Count null
df.isnull().sum()

# Handle categorical data
list1 = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
df[list1] = df[list1].replace({'yes':0,'no':1})

df['furnishingstatus'] = df['furnishingstatus'].replace({
    'furnished':0,
    'semi-furnished':1,
    'unfurnished':2
})

# Scaling
scaler = StandardScaler()
list2=['price','area']
df[list2]=scaler.fit_transform(df[list2])

# Visualization
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True)
plt.show()

# Probability Distribution
df.hist(figsize=(10,10),bins=10)
plt.show

# Feature & target
X = df.drop('price', axis=1)
y = df['price']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

