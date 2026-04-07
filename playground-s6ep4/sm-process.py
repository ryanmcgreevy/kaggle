import argparse
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.2)
    parser.add_argument("--input-csv", type=str)
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))

    input_data_path = os.path.join("/opt/ml/processing/input", args.input_csv)

    print("Reading input data from {}".format(input_data_path))

    testing_mode = True
    balanced = True
    df_tv = pd.read_csv(input_data_path)

  
    print("Running preprocessing and feature engineering transformations")

    df_x = df_tv.iloc[:,1:-1]
    
    df_dummy = pd.get_dummies(df_x, dtype=int, drop_first=False)
    continous_variables = df_dummy.select_dtypes(['float64']).columns
    index = [df_dummy.columns.get_loc(col) for col in continous_variables]

    x = df_dummy.iloc[:,:].values
    #newcolumns = x.columns.values.tolist() + ['Irrigation_Need']
    y = df_tv.iloc[:,-1].values

    class_le = LabelEncoder()
    y = class_le.fit_transform(y)

    if testing_mode:
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = \
            train_test_split(x, y, 
                            test_size=0.20,
                            stratify=y,
                            random_state=1)
    else:
        X_train, y_train = x, y

    sc = StandardScaler().fit(X_train[:, index])
    X_train[:, index] = sc.transform(X_train[:, index])

    if testing_mode:
        X_test[:, index] = sc.transform(X_test[:, index])


    if balanced:
        majority_class = np.argmax(np.bincount(y_train))
        minority_class = np.argmin(np.bincount(y_train))
        middle_class = list(set(np.unique(y_train)) - set([majority_class, minority_class]))[0]
        X_train_majority = X_train[y_train == majority_class]
        y_train_majority = y_train[y_train == majority_class]
        
        X_train_minority = X_train[y_train == minority_class]
        y_train_minority = y_train[y_train == minority_class]
        
        X_train_middle = X_train[y_train == middle_class]
        y_train_middle = y_train[y_train == middle_class]
        
        
        X_train_minority_upsampled, y_train_minority_upsampled = resample(X_train_minority, y_train_minority,
                                                                        replace=True,
                                                                        n_samples=X_train_middle.shape[0],
                                                                        random_state=1)
        X_train_majority_downsampled, y_train_majority_downsampled = resample(X_train_majority, y_train_majority,
                                                                        replace=False,
                                                                        n_samples=X_train_middle.shape[0],
                                                                        random_state=1)
        X_train_balanced = np.vstack((X_train_majority_downsampled, X_train_middle, X_train_minority_upsampled))
        y_train_balanced = np.hstack((y_train_majority_downsampled, y_train_middle, y_train_minority_upsampled))

        perm = np.random.permutation(len(X_train_balanced))

        X_train = X_train_balanced[perm]
        y_train = y_train_balanced[perm]

        #train_features = np.hstack((X_train, y_train.reshape(-1, 1)))
        #df_train = pd.DataFrame(train_features, columns=newcolumns)
    

    train_features = X_train
    test_features = X_test if testing_mode else None
    print("Train data shape after preprocessing: {}".format(train_features.shape))
    print("Test data shape after preprocessing: {}".format(test_features.shape))

    train_features_output_path = os.path.join("/opt/ml/processing/train", "train_features.csv")
    train_labels_output_path = os.path.join("/opt/ml/processing/train", "train_labels.csv")

    test_features_output_path = os.path.join("/opt/ml/processing/test", "test_features.csv")
    test_labels_output_path = os.path.join("/opt/ml/processing/test", "test_labels.csv")

    print("Saving training features to {}".format(train_features_output_path))
    pd.DataFrame(train_features).to_csv(train_features_output_path, header=False, index=False)

    print("Saving test features to {}".format(test_features_output_path))
    pd.DataFrame(test_features).to_csv(test_features_output_path, header=False, index=False)

    print("Saving training labels to {}".format(train_labels_output_path))
    pd.DataFrame(y_train).to_csv(train_labels_output_path, header=False, index=False)

    print("Saving test labels to {}".format(test_labels_output_path))
    pd.DataFrame(y_test).to_csv(test_labels_output_path, header=False, index=False)