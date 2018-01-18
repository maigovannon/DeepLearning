from Interface import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def L_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []

    params = initialize_parameters_random(layer_dims)

    for i in range(num_iterations):
        AL, caches = L_model_forward(X, params)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        params = update_parameters(params, grads, learning_rate)

        if i % 100 == 0:
            if print_cost:
                print("Cost after iteration %d: %f" % (i, cost))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per thousands)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return params


def clean_data(data_cleaner):
    for dataset in data_cleaner:
        dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
        dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
        dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)
    data1.drop(['Cabin', 'Ticket', 'PassengerId'], axis=1, inplace=True)

def feature_engineer (data_cleaner):
    for dataset in data_cleaner:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']
        dataset['IsAlone'] = (dataset['FamilySize'] == 0).astype(int)
        dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
        dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
        dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

def convert_formats(data_cleaner):
    label = preprocessing.LabelEncoder()
    for dataset in data_cleaner:
        dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
        dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
        dataset['Title_Code'] = label.fit_transform(dataset['Title'])
        dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
        dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

train_file = r"C:\developer\kaggle\titanic\train.csv"
test_file = r"C:\developer\kaggle\titanic\test.csv"
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
data1 = train_data.copy()

clean_data([data1, test_data])
feature_engineer([data1, test_data])
convert_formats([data1, test_data])

ipLabels = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
opLabels = ['Survived']
X = data1[ipLabels]
# X = data1.loc[:, data1.columns != 'Survived']
Y = data1[opLabels]
df.to_csv(X, 'foo.csv')

trainX, testX, trainY, testY = train_test_split(X, Y, random_state=0)

layers_dims = [X.shape[1], 100, 20,  1]
params = L_layer_model(trainX.T.values, trainY.T.values, layers_dims, print_cost=True, num_iterations=35000, learning_rate=0.0075)
acc = accuracy(trainX.T.values, trainY.T.values, params)
print(f"Train accuracy = {acc}%")
print(f"Test accuracy = {accuracy(testX.T.values, testY.T.values, params)}")

