import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
import itertools
import operator

def read_data():
    data = []
    with open('transfusion.data') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # take the header out
        for row in reader:  # each row is a list
            data.append(row)
    data = np.array(data, dtype=np.int32)
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


X, y = read_data()

num_of_rows = len(X)
total_num_of_folds = 5
num_of_folds_in_train_data = 4
num_of_folds_in_test_data = 1

#First, we split data into 5 folds
x_split_in_folds = np.array_split(X, total_num_of_folds)
y_split_in_folds = np.array_split(y, total_num_of_folds)

f1_scores_on_test = []
#Pick which fold to allocate as test data
for test_fold_pos in range(total_num_of_folds):
    x_test_data = x_split_in_folds[test_fold_pos]
    y_test_data = y_split_in_folds[test_fold_pos]

    #Remaining folds constitute training data
    if test_fold_pos == 0:
        x_train_data_folds = x_split_in_folds[1:]
        y_train_data_folds = y_split_in_folds[1:]

    elif test_fold_pos == total_num_of_folds - 1:
        x_train_data_folds = x_split_in_folds[:-1]
        y_train_data_folds = y_split_in_folds[:-1]

    else:
        x_train_data_folds = x_split_in_folds[:test_fold_pos] + x_split_in_folds[test_fold_pos + 1:]
        y_train_data_folds = y_split_in_folds[:test_fold_pos] + y_split_in_folds[test_fold_pos + 1:]

    c_performance_averages = dict()
    #Now, specify the validation fold - starting from the first fold of the total training data - and build model
    for validation_pos in range(len(x_train_data_folds)):
        x_validation_data = x_train_data_folds[validation_pos]
        y_validation_data = y_train_data_folds[validation_pos]

        #Combine remaining train data as a single list
        if validation_pos == 0:
            x_train_data_remaining_folds = x_train_data_folds[1:]
            y_train_data_remaining_folds = y_train_data_folds[1:]

        elif validation_pos == len(x_train_data_folds) - 1:
            x_train_data_remaining_folds = x_train_data_folds[:-1]
            y_train_data_remaining_folds = y_train_data_folds[:-1]

        else:
            x_train_data_remaining_folds = x_train_data_folds[:validation_pos] + x_train_data_folds[validation_pos + 1:]
            y_train_data_remaining_folds = y_train_data_folds[:validation_pos] + y_train_data_folds[validation_pos + 1:]

        x_train_data_remaining = list(itertools.chain(*x_train_data_remaining_folds))
        y_train_data_remaining = list(itertools.chain(*y_train_data_remaining_folds))

        #Select hyperparameter to train on
        C_values = [0.1, 1, 10, 100]
        for C in C_values:
            model = LogisticRegression(C=C, solver='lbfgs')
            model.fit(x_train_data_remaining, y_train_data_remaining)  # training
            y_pred = model.predict(x_validation_data)  # predicting
            f1_value = f1_score(y_validation_data, y_pred)
            if C in c_performance_averages:
                c_performance_averages[C] = c_performance_averages[C] + f1_value
            else:
                c_performance_averages[C] = f1_value

    #Calculate average for each value of C
    for param in c_performance_averages.keys():
        c_performance_averages[param] = c_performance_averages[param]/len(x_train_data_folds)
    highest_performing_c = max(c_performance_averages.items(), key=operator.itemgetter(1))[0]

    print("Test Data Pos - " +str(test_fold_pos) + "\t Highest C - " +str(highest_performing_c))

    #Use model of highest-C on test data
    x_all_training_data = list(itertools.chain(*x_train_data_folds))
    y_all_training_data = list(itertools.chain(*y_train_data_folds))
    model_for_test = LogisticRegression(C=highest_performing_c) # predicting
    model.fit(x_all_training_data,y_all_training_data)
    y_pred = model.predict(x_test_data)  # predicting
    f1_value = f1_score(y_test_data, y_pred)
    print("F1 Value on Test Data - " +str(f1_value))
    f1_scores_on_test.append(f1_value)

#Reporting average scores
average_f1_score_evaluation = np.average(f1_scores_on_test)
standard_deviation_f1_score_evaluation = np.std(f1_scores_on_test)
print("Average - " +str(average_f1_score_evaluation))
print("Standard Dev - " + str(standard_deviation_f1_score_evaluation))