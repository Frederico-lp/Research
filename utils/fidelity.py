from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

def eval_fidelity(X_train, y_train, X_test, X_syn_train, X_syn_test, y_syn_train):
    learners = [(AdaBoostClassifier(n_estimators=50))]

    history = dict()

    for i in range(len(learners)):
        model_real = learners[i]
        model_real.fit(X_train, y_train)

        model_fake = learners[i]
        model_fake.fit(X_syn_train, y_syn_train)

        #first letter is data where it trained, second is data where it tested
        rr_pred = []
        fr_pred = []
        ff_pred = []
        rf_pred = []

        # print(len(X_test))
        # print(len(X_syn_test))

        # if len(X_test != len(X_syn_test)):
        #     print("stop")
        #     break

        for j in range (len(X_test)):
            rr_pred.append(model_real.predict(X_test.iloc[j]))
            fr_pred.append(model_fake.predict(X_test.iloc[j]))
            rf_pred.append(model_fake.predict(X_syn_test.iloc[j]))
            ff_pred.append(model_fake.predict(X_syn_test.iloc[j]))
        
    return rr_pred, fr_pred, ff_pred, rf_pred

        # for index, row in X_test.iterrows():
        #     rr_pred.append(model_real.predict(row))
        #     fr_pred.append(model_fake.predict(row))
        #     # rf_pred.append(model_fake.predict(X_syn_test[j]))
        #     # ff_pred.append(model_fake.predict(X_syn_test[j]))
        
        # for index, row in X_syn_test.iterrows():
        #     rf_pred.append(model_real.predict(row))
        #     ff_pred.append(model_fake.predict(row))
        


