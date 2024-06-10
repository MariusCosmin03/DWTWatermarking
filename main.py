from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import random
import pywt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from watermarking import embedWatermark, extractWatermark, verifyWatermark
from attacks import *



KEY_WATERMARK = 271124
KEY_SPLIT = 274853

ATTACK_KEY = 13059

DWT_MAX_LEVEL = 3
DATASET = 1


######              DATA LOADING                ##################
def  find_lowest_variance(data):
    m_ind = 0
    m_var = float("inf")
    for i in np.arange(len(data[0])):
        col = data[:, i]
        i_var = np.var(col)
        if(i_var < m_var):
            m_var = i_var
            m_ind = i
    return m_ind

def load_data(mode = 1):
    match mode:
        case 0:
            NR_OF_SPLITS = 10
            MARKED_DATA = 0.3

            data_iris, target_iris = load_iris(return_X_y=True)
            min_var = np.var(data_iris[:, 0])
            for i in np.arange(1, 4):
                if np.var(data_iris[:, i]) < min_var:
                    marked_attribute = i
                    min_var = np.var(data_iris[:, i])
            return data_iris, target_iris, marked_attribute, NR_OF_SPLITS, MARKED_DATA
        case 1:
            NR_OF_SPLITS = 50
            MARKED_DATA = 0.5

            data = pd.read_excel('data/Dry_Bean_Dataset.xlsx')
            l1=LabelEncoder()
            data["Class"]=l1.fit_transform(data["Class"])

            X=data.iloc[:,:-1].values 
            Y=data.iloc[:,-1].values 

            marked_attribute = find_lowest_variance(X)
            return X, Y, marked_attribute, NR_OF_SPLITS, MARKED_DATA
        case _:
            raise Exception("Select a valid dataset to load")


X, Y, marked_attribute, NR_OF_SPLITS, MARKED_DATA = load_data(DATASET)
# print(marked_attribute)
x = X[:, marked_attribute]

###############################################################################################################################################################

def embedMultipleTimes(NR_OF_SPLITS, r_s):
    
    fig, axs = plt.subplots(3)
    axs[0].plot(x)

    SELECTED_SPLITS = int(MARKED_DATA * NR_OF_SPLITS)
    x_WM = embedWatermark(x, KEY_WATERMARK, KEY_SPLIT, SELECTED_SPLITS, NR_OF_SPLITS, DWT_MAX_LEVEL)
    axs[1].plot(x_WM)
    axs[2].plot(x - x_WM)
    plt.show()
    
    # nr = len(y)
    # print(nr)
    
    
    # z = removal_attack(y, 0.05, r_s)
    # z = add_attack(X, y, marked_attribute, 0.01, r_s)
    # print(np.mean(z))

    res = extractWatermark(x_WM, KEY_SPLIT, SELECTED_SPLITS, NR_OF_SPLITS, DWT_MAX_LEVEL)
    
    # print("Watermark is extracted correctly: " + str(verifyWatermark(KEY_WATERMARK, SELECTED_SPLITS, res)))

    # print("Extracted watermark: \n" + str(res))

    print("Mean of original data: ", np.mean(x))
    print("Mean of watermarked data: ", np.mean(y))
    print("Difference in mean: ", np.mean(x) - np.mean(y))
    print("% Difference in mean: ", (np.mean(y)-np.mean(x)) / np.mean(x) * 100)



    print("Variance of original data: " + str(np.var(x)))
    print("Variance of watermarked data: " + str(np.var(y)))
    print("Difference in variance: " + str(np.var(x) - np.var(y)))
    print("% Difference in variance: ", (np.var(y)-np.var(x)) / np.var(x) * 100)

    # print("Number of elements in initial data: " + str(len(x)))
    # print("Number of elements in watermarked data: " + str(len(y)))

    return verifyWatermark(KEY_WATERMARK, SELECTED_SPLITS, res)

def changesInDataAfterWM(x):

    q = pywt.dwt_max_level(len(x)/(NR_OF_SPLITS * 2), 'db1')

    if q > DWT_MAX_LEVEL :
        DWT_level = DWT_MAX_LEVEL
    elif q > 0 :
        DWT_level = q - 1
    else :
        DWT_level = q
    print("Max DWT level: " + str(q), DWT_level)

    fig, axs = plt.subplots(3)
    axs[0].plot(x)
    axs[0].title.set_text("Original Data")

    SELECTED_SPLITS = int(MARKED_DATA * NR_OF_SPLITS)
    x_WM = embedWatermark(x, KEY_WATERMARK, KEY_SPLIT, SELECTED_SPLITS, NR_OF_SPLITS, DWT_level)

    axs[1].plot(x_WM)
    axs[1].title.set_text("Watermaked Data")
    axs[2].plot(x - x_WM)
    axs[2].title.set_text("Difference induced by watermarking")

    fig.tight_layout()
    plt.show()
    
    # print(y.size)

    res = extractWatermark(x_WM, KEY_SPLIT, SELECTED_SPLITS, NR_OF_SPLITS, DWT_level)

    # print("Extracted watermark: \n" + str(res))

    print("Mean of original data: ", f"{np.mean(x):.6e}")
    print("Mean of watermarked data: ", f"{np.mean(x_WM):.6e}")
    print("Difference in mean: ", f"{np.mean(x_WM) - np.mean(x):.6e}")
    print("% Difference in mean: ", f"{(np.mean(x_WM)-np.mean(x)) / np.mean(x) * 100:.4f}")



    print("Variance of original data: ", f"{(np.var(x)):.6e}")
    print("Variance of watermarked data: ", f"{np.var(x_WM):.6e}" )
    print("Difference in variance: ", f"{np.var(x_WM) - np.var(x):.6e}")
    print("% Difference in variance:", f"{(np.var(x_WM)-np.var(x)) / np.var(x) * 100:.4f}")

    # print("Number of elements in initial data: " + str(len(x)))
    # print("Number of elements in watermarked data: " + str(len(y)))

    print("Watermark was extracted succesfuly:", verifyWatermark(KEY_WATERMARK, SELECTED_SPLITS, res))


changesInDataAfterWM(x)


def attackDataset(X, marked_attribute, fractions, typeOfAttack = 0, nrOfAttacks = 200, ATTACK_KEY = ATTACK_KEY):
    x = X[:, marked_attribute]

    q = pywt.dwt_max_level(len(x)/(NR_OF_SPLITS * 2), 'db1')

    if q > DWT_MAX_LEVEL :
        DWT_level = DWT_MAX_LEVEL
    elif q > 0 :
        DWT_level = q - 1
    else :
        DWT_level = q
    # print("Max DWT level: " + str(q))

    SELECTED_SPLITS = int(MARKED_DATA * NR_OF_SPLITS)
    x_WM = embedWatermark(x, KEY_WATERMARK, KEY_SPLIT, SELECTED_SPLITS, NR_OF_SPLITS, DWT_level)

    np.random.seed(ATTACK_KEY)
    attackKeys = np.random.choice(range(100000), size=nrOfAttacks, replace=False)


    match typeOfAttack:
        case 0:
            res = np.array([])
            for fraction in fractions:
                total = 0
                for i in np.arange(nrOfAttacks):
                    x_WM_attacked = update_attack_on_marked_attr(X, x_WM, marked_attribute, fraction, attackKeys[i])
                    total += verifyWatermark(KEY_WATERMARK, SELECTED_SPLITS, extractWatermark(x_WM_attacked, KEY_SPLIT, SELECTED_SPLITS, NR_OF_SPLITS, DWT_level))
                res = np.append(res, total)

                print("UPDATE ATTACK: Out of" , nrOfAttacks, ",", total, "watermarks were extracted correctly when", fraction, "of the data was attacked")
                print("UPDATE ATTACK: Precision:", f"{total/nrOfAttacks * 100 : .2f}")
                # print(x_WM_attacked.size)
            return res

        case 1:
            res = np.array([])
            for fraction in fractions:
                total = 0
                for i in np.arange(nrOfAttacks):
                    x_WM_attacked = removal_attack(x_WM, fraction, attackKeys[i])
                    total += verifyWatermark(KEY_WATERMARK, SELECTED_SPLITS, extractWatermark(x_WM_attacked, KEY_SPLIT, SELECTED_SPLITS, NR_OF_SPLITS, DWT_level))
                res = np.append(res, total)

                print("DELETE ATTACK: Out of" , nrOfAttacks, ",", total, "watermarks were extracted correctly when", fraction, "of the data was attacked")
                print("DELETE ATTACK: Precision:", f"{total/nrOfAttacks * 100 : .2f}")
                # print(x_WM.size)
                # print(x_WM_attacked.size)

            return res

        case 2:
            res = np.array([])
            for fraction in fractions:
                total = 0
                for i in np.arange(nrOfAttacks):
                    x_WM_attacked = zero_out_attack(x_WM, fraction, attackKeys[i])
                    total += verifyWatermark(KEY_WATERMARK, SELECTED_SPLITS, extractWatermark(x_WM_attacked, KEY_SPLIT, SELECTED_SPLITS, NR_OF_SPLITS, DWT_level))
                res = np.append(res, total)

                print("ZERO-OUT ATTACK: Out of" , nrOfAttacks, ",", total, "watermarks were extracted correctly when", fraction, "of the data was attacked")
                print("ZERO-OUT ATTACK: Precision:", f"{total/nrOfAttacks * 100 : .2f}")
                # print(x_WM_attacked.size)
            print(np.sum(x_WM))
            return res

        case 3:
            res = np.array([])
            for fraction in fractions:
                total = 0
                for i in np.arange(nrOfAttacks):
                    x_WM_attacked = add_attack(X, x_WM, marked_attribute, fraction, attackKeys[i])
                    total += verifyWatermark(KEY_WATERMARK, SELECTED_SPLITS, extractWatermark(x_WM_attacked, KEY_SPLIT, SELECTED_SPLITS, NR_OF_SPLITS, DWT_MAX_LEVEL))
                res = np.append(res, total)

                print("CREATE ATTACK: Out of" , nrOfAttacks, ",", total, "watermarks were extracted correctly when", fraction, "of the data was attacked")
                print("CREATE ATTACK: Precision:", f"{total/nrOfAttacks * 100 : .2f}")
                # print(x_WM_attacked.size)
            return res

        case _:
            raise Exception("Invalid attack type. Select a number in range 0-3")

fractions = [0, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1] 
# attackDataset(X, marked_attribute, fractions, 3, 200)

def drawAttacks(X, marked_attribute, fractions, nrOfAttacks, ATTACK_KEY = ATTACK_KEY):
    plt.plot(fractions, attackDataset(X, marked_attribute, fractions, 0, nrOfAttacks, ATTACK_KEY)/nrOfAttacks, color = "red", label = "Update Attack")
    plt.plot(fractions, attackDataset(X, marked_attribute, fractions, 1, nrOfAttacks, ATTACK_KEY)/nrOfAttacks, color = "blue", label = "Delete Attack")
    plt.plot(fractions, attackDataset(X, marked_attribute, fractions, 2, nrOfAttacks, ATTACK_KEY)/nrOfAttacks, color = "green", label = "Zero-Out Attack")
    plt.plot(fractions, attackDataset(X, marked_attribute, fractions, 3, nrOfAttacks, ATTACK_KEY)/nrOfAttacks, color = "m", label = "Create Attack")
    plt.legend()
    plt.show()

# drawAttacks(X, marked_attribute, fractions, 200)

############################################################################################                     MODEL TRAINING                       ################################################################################################################

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")

def WM_and_normalize_data(X, marked_attribute, selected_splits = 50, splits = 50):
    X_WM = X.copy()
    s_sc = StandardScaler()
    X = s_sc.fit_transform(X)
    
    x = X_WM[: , marked_attribute]
    z = embedWatermark(x, KEY_WATERMARK, KEY_SPLIT, selected_splits, splits, DWT_MAX_LEVEL)

    X_WM[:, marked_attribute] = z
    X_WM = s_sc.fit_transform(X_WM)
    return X, X_WM

def train_WM_logReg(X, Y, test_size, marked_attribute):

    X, X_WM = WM_and_normalize_data(X, marked_attribute)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    # Tune Hyperparams

    params = {"C": np.logspace(-4, 4, 20),
          "solver": ["liblinear", "saga"], 
          "max_iter": [10000],
          }
    lr_clf = LogisticRegression()

    lr_cv = GridSearchCV(lr_clf, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=5)
    lr_cv.fit(X_train, y_train)
    best_params = lr_cv.best_params_
    print(f"Best parameters: {best_params}")
    lr_clf = LogisticRegression(**best_params)

    
    lr_clf.fit(X_train, y_train)

    # print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
    # print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)


    X_train_WM, X_test_WM, y_train_WM, y_test_WM = train_test_split(X_WM, Y, test_size=test_size)


    lr_clf_WM = LogisticRegression()

    lr_cv_WM = GridSearchCV(lr_clf_WM, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=5)
    lr_cv_WM.fit(X_train, y_train)
    best_params_WM = lr_cv_WM.best_params_
    print(f"Best parameters: {best_params_WM}")
    lr_clf_WM = LogisticRegression(**best_params_WM)


    lr_clf_WM.fit(X_train_WM, y_train_WM)


    # print_score(lr_clf, X_train_WM, y_train_WM, X_test_WM, y_test_WM, train=True)
    # print_score(lr_clf_WM, X_train_WM, y_train_WM, X_test_WM, y_test_WM, train=False)

    return [accuracy_score(y_train, lr_clf.predict(X_train)) * 100, accuracy_score(y_test, lr_clf.predict(X_test)) * 100, accuracy_score(y_train_WM, lr_clf.predict(X_train_WM)) * 100, accuracy_score(y_test_WM, lr_clf.predict(X_test_WM)) * 100]

def train_WM_KNN(X, Y, test_size, marked_attribute):
    X, X_WM = WM_and_normalize_data(X, marked_attribute)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    knn_clf = KNeighborsClassifier(n_neighbors=6)
    knn_clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, knn_clf.predict(X_train)) * 100
    test_acc = accuracy_score(y_test, knn_clf.predict(X_test)) * 100


    X_train_WM, X_test_WM, y_train_WM, y_test_WM = train_test_split(X_WM, Y, test_size=test_size)

    knn_clf.fit(X_train_WM, y_train_WM)

    return [train_acc, test_acc, accuracy_score(y_train_WM, knn_clf.predict(X_train_WM)) * 100, accuracy_score(y_test_WM, knn_clf.predict(X_test_WM)) * 100]

def train_WM_SVM(X, Y, test_size, marked_attribute):
    X, X_WM = WM_and_normalize_data(X, marked_attribute)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    X_train_WM, X_test_WM, y_train_WM, y_test_WM = train_test_split(X_WM, Y, test_size=test_size)

    svm_clf = SVC(kernel='rbf', gamma=0.1, C=1.0)

    params = {"C":(0.1, 0.5, 1, 2, 5, 10, 20), 
            "gamma":(0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1), 
            "kernel":('linear', 'poly', 'rbf')}

    svm_cv = GridSearchCV(svm_clf, params, n_jobs=-1, cv=5, verbose=1, scoring="accuracy")
    svm_cv.fit(X_train, y_train)
    best_params = svm_cv.best_params_
    # print(f"Best params: {best_params}")
    svm_clf = SVC(**best_params)
    svm_clf.fit(X_train, y_train)

    

    svm_clf_WM = SVC(kernel='rbf', gamma=0.1, C=1.0)

    svm_cv_WM = GridSearchCV(svm_clf_WM, params, n_jobs=-1, cv=5, verbose=1, scoring="accuracy")
    svm_cv_WM.fit(X_train_WM, y_train_WM)
    best_params_WM = svm_cv_WM.best_params_
    # print(f"Best params: {best_params_WM}")
    svm_clf_WM = SVC(**best_params_WM)
    svm_clf_WM.fit(X_train_WM, y_train_WM)


    return [accuracy_score(y_train, svm_clf.predict(X_train)) * 100, accuracy_score(y_test, svm_clf.predict(X_test)) * 100, accuracy_score(y_train_WM, svm_clf_WM.predict(X_train_WM)) * 100, accuracy_score(y_test_WM, svm_clf_WM.predict(X_test_WM)) * 100]

def train_WM_DecTree(X, Y, test_size, marked_attribute):
    X, X_WM = WM_and_normalize_data(X, marked_attribute)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    X_train_WM, X_test_WM, y_train_WM, y_test_WM = train_test_split(X_WM, Y, test_size=test_size)
    params = {"criterion":("gini", "entropy"), 
            "splitter":("best", "random"), 
            "max_depth":(list(range(1, 20))), 
            "min_samples_split":[2, 3, 4], 
            "min_samples_leaf":list(range(1, 20))
            }

    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_cv = GridSearchCV(tree_clf, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=5)
    tree_cv.fit(X_train, y_train)
    best_params = tree_cv.best_params_
    print(f'Best_params: {best_params}')

    tree_clf = DecisionTreeClassifier(**best_params)
    tree_clf.fit(X_train, y_train)





    tree_clf_WM = DecisionTreeClassifier(random_state=42)
    tree_cv_WM = GridSearchCV(tree_clf_WM, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=5)
    tree_cv_WM.fit(X_train_WM, y_train_WM)
    best_params_WM = tree_cv_WM.best_params_
    print(f'Best_params: {best_params_WM}')

    tree_clf_WM = DecisionTreeClassifier(**best_params_WM)
    tree_clf_WM.fit(X_train_WM, y_train_WM)

    return [accuracy_score(y_train, tree_clf.predict(X_train)) * 100, accuracy_score(y_test, tree_clf.predict(X_test)) * 100, accuracy_score(y_train_WM, tree_clf_WM.predict(X_train_WM)) * 100, accuracy_score(y_test_WM, tree_clf_WM.predict(X_test_WM)) * 100]
    
def train_WM_RandomForest(X, Y, test_size, marked_attribute):
    X, X_WM = WM_and_normalize_data(X, marked_attribute)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    X_train_WM, X_test_WM, y_train_WM, y_test_WM = train_test_split(X_WM, Y, test_size=test_size)


    params_grid = {
        'n_estimators': [500, 900, 1100, 1500], 
        'max_features': ['sqrt', 'log2'],
        'max_depth': [2, 3, 5, 10, 15, None], 
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
                }


    rf_clf = RandomForestClassifier(random_state=42)
    rf_cv = GridSearchCV(rf_clf, params_grid, scoring="accuracy", cv=5, verbose=1, n_jobs=-1)
    rf_cv.fit(X_train, y_train)
    best_params = rf_cv.best_params_
    print(f"Best parameters: {best_params}")

    rf_clf = RandomForestClassifier(**best_params)
    rf_clf.fit(X_train, y_train)

    rf_clf_WM = RandomForestClassifier(random_state=42)
    rf_cv_WM = GridSearchCV(rf_clf_WM, params_grid, scoring="accuracy", cv=5, verbose=1, n_jobs=-1)
    rf_cv_WM.fit(X_train, y_train)
    best_params_WM = rf_cv_WM.best_params_
    print(f"Best parameters: {best_params_WM}")

    rf_clf_WM = RandomForestClassifier(**best_params_WM)
    rf_clf_WM.fit(X_train, y_train)

    return [accuracy_score(y_train, rf_clf.predict(X_train)) * 100, accuracy_score(y_test, rf_clf.predict(X_test)) * 100, accuracy_score(y_train_WM, rf_clf_WM.predict(X_train_WM)) * 100, accuracy_score(y_test_WM, rf_clf_WM.predict(X_test_WM)) * 100]


def drawKNN(X, Y, test_size, marked_attribute):
    res = []
    neighbours = np.arange(1, 30)
    for i in neighbours:
        res.append(train_WM_KNN(X, Y, test_size, marked_attribute, i)[0])
    # print(res)
    plt.figure(figsize=(10, 7))
    plt.plot(res, scaley=(80, 110))
    plt.xticks(np.arange(1, 31, 1))
    plt.show()

# drawKNN(X, Y, 0.3, marked_attribute)

def train_WM(X, Y, test_size, marked_attribute):



    tuning_results_df = pd.DataFrame(
        data=[# ["Tuned Logistic Regression"] + train_WM_logReg(X, Y, test_size, marked_attribute),
                ["KNN with 20 neightbours"] + train_WM_KNN(X, Y, test_size, marked_attribute),
                # ["Tuned Support Vector Machine"] + train_WM_SVM(X, Y, test_size, marked_attribute), 
                # ["Tuned Decision Tree"] + train_WM_DecTree(X, Y, test_size, marked_attribute), 
                # ["Tuned Random Forest"] + train_WM_RandomForest(X, Y, test_size, marked_attribute),
        ],
        columns=['Model', 'Training Accuracy %', 'Testing Accuracy %', 'Training Accuracy WM %', 'Testing Accuracy WM %']
    )

    return tuning_results_df

# print(train_WM(X, Y, 0.3, marked_attribute))
