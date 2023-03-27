from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

def get_predictions(X_train, y_train, X_test, y_test):
    learners = [(AdaBoostClassifier(n_estimators=50))]
    #learners = [(RandomForestClassifier())]

    history = dict()

    for i in range(len(learners)):
        model = learners[i]
        model.fit(X_train, y_train)

        #first letter is data where it trained, second is data where it tested
        pred = []

        for j in range (len(X_test)):
            #print(X_test.loc[[j]])
            pred.append(model.predict(X_test.iloc[[j]]))
        
    return pred
