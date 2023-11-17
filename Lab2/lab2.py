import warnings
with warnings.catch_warnings():
    # filter sklearn\externals\joblib\parallel.py:268:
    # DeprecationWarning: check_pickle is deprecated
    warnings.simplefilter("ignore", category=FutureWarning)
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier

def main():
    data = pd.read_csv("Face-Feat.txt")
    print(data)
    labels = data["Emotion"]
    inputs = data.drop("Emotion", axis = 1)
    data_in, test_in, data_out, test_out = train_test_split(inputs, labels, test_size=0.1, stratify=labels)
    train_in, val_in, train_out, val_out = train_test_split(data_in, data_out, test_size = 0.2/0.9, stratify = data_out)
    model = SVC()
    model.fit(train_in, train_out)
    output = model.predict(val_in)
    print("Val accuracy is ", accuracy_score(val_out, output)*100)

    model_2 = DecisionTreeClassifier()
    model_2.fit(train_in, train_out)
    output = model_2.predict(val_in)
    print("Val accuracy is ", accuracy_score(val_out, output)*100)


    param_grid=[
        {"kernel": ["linear","poly","rbf"]},
        {"kernel": ["poly"], "degree": [2, 5, 15]}
    ]
    meta_model = GridSearchCV(SVC(), param_grid=param_grid)
    meta_model.fit(train_in, train_out)
    output = meta_model.predict(test_in)
    print("Val accuracy is ", accuracy_score(val_out, output)*100)
    print(meta_model.best_params_)

    #bayesian
    #lock 

if __name__ == "__main__":
    main()