import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


def main_logic():
    
    df = pd.read_csv('bank-additional-full.csv', delimiter=';')
    print('Columns are', df.columns.values)
    print(df.dtypes)
    
    relevant_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                        'contact', 'month', 'day_of_week', 'poutcome']

    for rc in relevant_columns:
        print("Column '{rc}': {values}".format(rc=rc, values=df[rc].unique()))
    
    df['class'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)
    
    df_new = pd.get_dummies(df, columns=relevant_columns)
    print('New dataset columns are', df_new.columns.values)
    
    print('===================================================')
    print('# of total records =', df.shape[0])
    print('# of success samples', sum(df['class'] == 1))
    print('# of failed samples', sum(df['class'] == 0))
    print('===================================================')
    
    x_all = df_new.drop(['y', 'class'], axis=1)
    y_all = df['class']
    
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42)

    print('===================================================')
    print('# of training records', x_train.shape[0])
    print('# of testing records', x_test.shape[0])
    print('===================================================')
    
    classifier = LogisticRegression(penalty='l2', C=0.001, class_weight="balanced")
    classifier.fit(x_train, y_train)
    y_predict = classifier.predict(x_test)
    
    pickle.dump(classifier, open('classifier.pkl', 'wb'))
    
    def print_scores(actual, prediction):
        print("accuracy is %f" % accuracy_score(actual, prediction))
        print("precision is %f" % precision_score(actual, prediction))
        print("f1 is %f" % f1_score(actual, prediction))
        print("recall is %f" % recall_score(actual, prediction))
        print('Confusion Matrix:')
        print(confusion_matrix(actual, prediction))
        
    classifier = DecisionTreeClassifier(criterion="gini", max_depth=6, min_samples_leaf=16, min_samples_split=16)
    classifier.fit(x_train, y_train)
    y_predict = classifier.predict(x_test)
    print('\nPerformance metrics for Decision Tree Classifier:')
    print_scores(y_test, y_predict)
        
    
if __name__ == "__main__":
    main_logic()