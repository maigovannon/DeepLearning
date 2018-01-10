from Interface import *
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df


def L_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []

    params = initialize_parameters_deep(layer_dims)

    for i in range(num_iterations):
        AL, caches = L_model_forward(X, params)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        params = update_parameters(params, grads, learning_rate)

        if print_cost:
            if i % 100:
                print("Cost after iteration {}: {}" % (i, cost))
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
        dataset['Embarked'].fillna(dataset['Embarked'].mode(), inplace=True)
        dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)
    data1.drop(['Cabin', 'Ticket', 'PassengerId'], axis=1, inplace=True)

def feature_engineer (data_cleaner):
    for dataset in data_cleaner:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']
        dataset['IsAlone'] = (dataset['FamilySize'] == 0).astype(int)
        dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
        dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)


train_file = r"C:\developer\titanic\train.csv"
test_file = r"C:\developer\titanic\test.csv"
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
data1 = train_data.copy()

clean_data([data1, test_data])
feature_engineer([data1, test_data])
print (data1.isnull().sum())
print(test_data.isnull().sum())

