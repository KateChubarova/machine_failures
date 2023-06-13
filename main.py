import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("train.csv")
X_submission = pd.read_csv('test.csv')

_data = data.drop(['id'], axis='columns')

print(_data.head())
print(_data.columns.values)


def corr():
    correlation_matrix = _data.corr()
    plt.figure(figsize=(12, 7))
    sns.heatmap(correlation_matrix, annot=True)
    plt.show()


def show_outliners():
    fig, axs = plt.subplots(ncols=len(_data.columns), figsize=(20, 6))
    for i, col in enumerate(_data.columns):
        sns.boxplot(data=_data[col], ax=axs[i])
        axs[i].set_title(col)
    plt.show()


def remove_outliers(train):
    train = train.copy()
    train = train.drop(train[train['Torque [Nm]'] > 50].index)
    train = train.drop(train[train['Torque [Nm]'] < 30].index)
    return train


_data = _data.drop(['Air temperature [K]', 'Process temperature [K]',
                    'Rotational speed [rpm]', 'RNF', 'Tool wear [min]'],
                   axis='columns')

_data = pd.get_dummies(_data, columns=['Type'])
_data = _data.drop(['Product ID'], axis='columns')

print(_data.info())
_data = remove_outliers(_data)

y = _data['Machine failure']
X = _data.drop(['Machine failure'], axis='columns')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


def write_to_file(predictions, X_submission):
    output = pd.DataFrame({'id': X_submission.id, 'Machine failure': predictions.reshape(len(predictions))})
    output.to_csv('submission.csv', index=False)


def tree_classifier(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=1)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

    _X_submission = X_submission.drop(['Air temperature [K]', 'Process temperature [K]',
                                       'Rotational speed [rpm]', 'RNF', 'Tool wear [min]', 'id', 'Product ID'],
                                      axis='columns')
    _X_submission = pd.get_dummies(_X_submission, columns=['Type'])
    y_submission = model.predict(_X_submission)
    print('tree_classifier')
    print(y_submission)
    write_to_file(y_submission, X_submission)


tree_classifier(X_train, y_train, X_test, y_test)
