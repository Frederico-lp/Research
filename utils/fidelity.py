import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, roc_auc_score

def get_predictions(X_train, y_train, X_test, y_test, undersample = True):
    if undersample:
        sampling_strategy = 0.75
        rus = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)
        X_train, y_train = rus.fit_resample(X_train, y_train) # type: ignore
        #X_test, y_test = rus.fit_resample(X_test, y_test)


        
    learners = [(AdaBoostClassifier(n_estimators=50))]
    #learners = [(RandomForestClassifier())]

    history = dict()

    for i in range(len(learners)):
        model = learners[i]
        model.fit(X_train, y_train)

        pred = []

        for j in range (len(X_test)):
            #print(X_test.loc[[j]])
            pred.append(model.predict(X_test.iloc[[j]]))
        
    return pred

def eval_fidelity(pred1, pred2):
    
    values = np.array(pred1)
    values = np.unique(values)
    class1 = values[0]
    class2 = values[1]
    class1_same = 0
    class1_dif = 0
    class2_same = 0
    class2_dif = 0

    same_pred = 0
    dif_pred = 0
    if len(pred1) != len(pred2):
        print("Error: different sizes")
    
    for i in range(len(pred1)):
        if pred1[i] == pred2[i]:
            same_pred += 1
            if pred1[i] == class1:
                class1_same += 1
            else:
                class2_same += 1

        else:
            dif_pred += 1
            class1_dif += 1
            class2_dif += 1

    ratio = same_pred / (same_pred + dif_pred)
    class1_ratio = class1_same / (class1_same + class1_dif)
    class2_ratio = class2_same / (class2_same + class2_dif)

    return ratio, class1_ratio, class2_ratio


def get_accuracy(y, pred):
    return accuracy_score(y, pred)

def get_roc_auc(y, pred):
    return roc_auc_score(y, pred)
    

