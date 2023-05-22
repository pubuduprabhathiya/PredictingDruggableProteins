# # Feature extractor
from propy import PyPro
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, f_classif
import optuna
import ast
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, matthews_corrcoef, recall_score, precision_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier


import sys

data_dir = 'dataset'

def extractProteinSequenceFromFasta(file):
    # read text file in FASTA format
    with open(file, 'r') as f:
        lines = f.readlines()
    # remove new line characters
    lines = [line.strip() for line in lines]
    # remove empty lines
    lines = [line for line in lines if line != '']
    # odd ids are protein sequences
    protein_sequences = lines[1::2]
    # even ids are protein ids
    protein_ids = lines[::2]
    # return protein sequences
    return protein_ids, protein_sequences

# function to extract AAC features from a given FASTA format txt file using propy3
def extractFeatureDF(protein_ids, protein_sequences, feature_type, negative):
    df = pd.DataFrame()
    # iterate over protein sequences
    for i in range(len(protein_sequences)):
        try:
            # get protein sequence
            protein = PyPro.GetProDes(protein_sequences[i])
            if feature_type == 'AAC':
                extractedFeatures = protein.GetAAComp()
            elif feature_type == 'APAAC':
                extractedFeatures = protein.GetAPAAC()
            elif feature_type == 'CTD':
                extractedFeatures = protein.GetCTD()
            elif feature_type == 'PAAC':
                extractedFeatures = protein.GetPAAC()
            elif feature_type == 'DPC':
                extractedFeatures = protein.GetDPComp()
            # convert dictionary to pandas dataframe
            df1 = pd.DataFrame.from_dict(extractedFeatures, orient='index').transpose()
            df1['id'] = protein_ids[i][1:]
            # add dataframe to main dataframe with df.concat
            df = pd.concat([df, df1], ignore_index=True)
            print(feature_type, f"Extracted features for sequence {i}", negative)
        except ZeroDivisionError:
            print(f"Skipping sequence {i} due to ZeroDivisionError")
            continue
    if negative:
        df['label'] = 0
    else:
        df['label'] = 1
    # return AAC features dataframe
    return df

def combineNegativeAndPositiveDFs(negativeFile, positiveFile, feature_type):
    # extract protein ids and sequences from negative FASTA file
    negative_ids, negative_sequences = extractProteinSequenceFromFasta(negativeFile)
    # extract protein ids and sequences from positive FASTA file
    positive_ids, positive_sequences = extractProteinSequenceFromFasta(positiveFile)
    # extract feature_type from negative FASTA file
    negativeDF = extractFeatureDF(negative_ids, negative_sequences, feature_type, negative=True)
    # extract feature_type from positive FASTA file
    positiveDF = extractFeatureDF(positive_ids, positive_sequences, feature_type, negative=False)
    # combine positive and negative dataframes
    combinedDF = pd.concat([negativeDF, positiveDF], ignore_index=True)
    # shuffle dataframe
    combinedDF = combinedDF.sample(frac=1).reset_index(drop=True)
    # return combined dataframe
    return combinedDF

def CreateDataset(positive_train_data, negative_train_data, positive_test_data, negative_test_data):
    combineNegativeAndPositiveDFs(negative_train_data, positive_train_data, 'AAC').to_csv(f'processed_dataset/TR_AAC.csv', index=False)
    combineNegativeAndPositiveDFs(negative_train_data, positive_train_data, 'APAAC').to_csv(f'processed_dataset/TR_APAAC.csv', index=False)
    combineNegativeAndPositiveDFs(negative_train_data, positive_train_data, 'CTD').to_csv(f'processed_dataset/TR_CTD.csv', index=False)
    combineNegativeAndPositiveDFs(negative_train_data, positive_train_data, 'PAAC').to_csv(f'processed_dataset/TR_PAAC.csv', index=False)
    combineNegativeAndPositiveDFs(negative_train_data, positive_train_data, 'DPC').to_csv(f'processed_dataset/TR_DPC.csv', index=False)

    combineNegativeAndPositiveDFs(negative_test_data, positive_test_data, 'AAC').to_csv(f'processed_dataset/TS_AAC.csv', index=False)
    combineNegativeAndPositiveDFs(negative_test_data, positive_test_data, 'APAAC').to_csv(f'processed_dataset/TS_APAAC.csv', index=False)
    combineNegativeAndPositiveDFs(negative_test_data, positive_test_data, 'CTD').to_csv(f'processed_dataset/TS_CTD.csv', index=False)
    combineNegativeAndPositiveDFs(negative_test_data, positive_test_data, 'PAAC').to_csv(f'processed_dataset/TS_PAAC.csv', index=False)
    combineNegativeAndPositiveDFs(negative_test_data, positive_test_data, 'DPC').to_csv(f'processed_dataset/TS_DPC.csv', index=False)
# # # classifiers-with-feature-combinations
def CreateModels():
    feature_types = ['AAC', 'APAAC', 'CTD', 'DPC', 'PAAC']

    data_dir = 'processed_dataset'
    feature_engineered_data_dir = 'output'

    # Concatenate features from different types
    train_dataframes = []
    test_dataframes = []

    for feature_type in feature_types:
        train_data = pd.read_csv(f'{data_dir}/TR_{feature_type}.csv')
        test_data = pd.read_csv(f'{data_dir}/TS_{feature_type}.csv')
        train_dataframes.append(train_data.set_index('id'))
        test_dataframes.append(test_data.set_index('id'))

    # concat dataframes based on equal column ids
    df_train = pd.concat(train_dataframes, axis=1, join='inner')
    df_test = pd.concat(test_dataframes, axis=1, join='inner')
    X_train = df_train.drop(columns=['label'], axis=1)
    y_train = df_train['label'].loc[:, ~df_train['label'].columns.duplicated()]

    def SelectFeatures(k):
        # Initialize SelectKBest class
        selector = SelectKBest(f_classif, k=k)

        # Fit and transform the data
        X_train_new = selector.fit_transform(X_train, y_train)

        # Get columns to keep
        cols = selector.get_support(indices=True)

        # Create new dataframe with only desired columns, or overwrite existing
        selected_features_df = X_train.iloc[:,cols]
        selected_features_df['label'] = y_train
        selected_features_df.reset_index(inplace=True)
        df_test_new=df_test
        df_test_new.reset_index(inplace=True)
        df_test_new = df_test_new[selected_features_df.columns].loc[:, ~df_test_new[selected_features_df.columns].columns.duplicated()]

        # Save the dataframe
        selected_features_df.to_csv(f'{feature_engineered_data_dir}/TR_selected_features_all_best{k}.csv', index=False)
        df_test_new.to_csv(f'{feature_engineered_data_dir}/TS_selected_features_all_best{k}.csv', index=False)
    NumofFea=[20,50,100]
    for num in NumofFea:
        SelectFeatures(num)

    data_dir = 'processed_dataset'
    feature_engineered_data_dir = 'output'

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)


    def evaluate_model(name, model, X_train, y_train, X_test, y_test, results_dataframe, feature_type):
        # evaluate model
        scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
        accuracy = scores.mean()

        # fit the model on the training set
        model.fit(X_train, y_train)

        # predict the test set results
        y_pred = model.predict(X_test)

        # compute the confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # calculate precision, recall (sensitivity), f1-score
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # calculate specificity
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn+fp)

        # calculate MCC
        mcc = matthews_corrcoef(y_test, y_pred)

        temp_df = pd.DataFrame({
            'feature_type': feature_type, 
            'model': name, 
            'with_hypertuning': False,
            'best_params': 'None',
            'accuracy': accuracy, 
            'sensitivity': recall, 
            'specificity': specificity, 
            'precision': precision, 
            'f1': f1, 
            'mcc': mcc,
            'index': f'{feature_type}_{name}_no_hypertuning'
            }, index=['index'])
        # results_dataframe is an empty dataframe to store results with the columns feature_type, model, with_hypertuning, accuracy, sensitivity, specificity, precision, f1, mcc
        return pd.concat([results_dataframe, temp_df])


    def optimize_hyperparameters(name, model, objective, trials, results_dataframe, feature_type, X_train, y_train, X_test, y_test):
        def optuna_objective(trial):
            params = objective(trial)
            model_instance = model(**params)
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)

            # compute the confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # calculate precision, recall (sensitivity), f1-score
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # calculate specificity
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn+fp)

            # calculate MCC
            mcc = matthews_corrcoef(y_test, y_pred)

            accuracy = (tp + tn) / (tp + tn + fp + fn)

            # Set user attributes
            trial.set_user_attr("precision", precision)
            trial.set_user_attr("recall", recall)
            trial.set_user_attr("f1", f1)
            trial.set_user_attr("specificity", specificity)
            trial.set_user_attr("mcc", mcc)

            return accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(optuna_objective, n_trials=trials)

        temp_df = pd.DataFrame({
            'feature_type': feature_type, 
            'model': name, 
            'with_hypertuning': True,
            'best_params': [str(study.best_trial.params)],
            'accuracy': study.best_trial.value, 
            'sensitivity': study.best_trial.user_attrs['recall'], 
            'specificity': study.best_trial.user_attrs['specificity'], 
            'precision': study.best_trial.user_attrs['precision'], 
            'f1': study.best_trial.user_attrs['f1'], 
            'mcc': study.best_trial.user_attrs['mcc'],
            'index': f'{feature_type}_{name}_with_hypertuning'
            }, index=['index'])
        results_dataframe = pd.concat([results_dataframe, temp_df])
        return results_dataframe


    # Define models
    models = {
        'LogisticRegression': LogisticRegression(),
        'SVC': SVC(),
        'XGBClassifier': XGBClassifier(),
        'LGBMClassifier': LGBMClassifier()
    }

    models_ = {
        'LogisticRegression': LogisticRegression,
        'SVC': SVC,
        'XGBClassifier': XGBClassifier,
        'LGBMClassifier': LGBMClassifier
    }

    # Define objectives for hyperparameters tuning
    objectives = {
        'LogisticRegression': lambda trial: {
            'C': trial.suggest_float('C', 1e-2, 1e-1),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
            'max_iter': trial.suggest_int('max_iter', 100, 1000)
        },
        'SVC': lambda trial: {
            'C': trial.suggest_float('svc_c', 1e-2, 1e2),
            'gamma': trial.suggest_float('svc_gamma', 1e-2, 1e2),
        },
        'XGBClassifier': lambda trial: {
            'learning_rate': trial.suggest_float("learning_rate", 1e-2, 0.3),
            'max_depth': trial.suggest_int("max_depth", 2, 6),
            'n_estimators': trial.suggest_int("n_estimators", 100, 1000)
        },
        'LGBMClassifier': lambda trial: {
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'max_depth': trial.suggest_int('max_depth', 2, 50),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000)
        }
    }

    """# Without Feature Selection"""

    # empty dataframe to store results with the columns feature_type, model, with_hypertuning, accuracy, sensitivity, specificity, precision, f1, mcc
    results = pd.DataFrame(columns=['feature_type', 'model', 'with_hypertuning', 'best_params', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'mcc', 'index'])
    feature_types = ['AAC', 'APAAC', 'CTD', 'DPC', 'PAAC']
    for feature_type in feature_types:

        # Load the training dataset
        data = pd.read_csv(f'{data_dir}/TR_{feature_type}.csv')

        # Separate features and target
        X = data.drop(columns=['label', 'id'], axis=1)
        y = data['label']

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Evaluate models without hyperparameters tuning
        for name, model in models.items():
            print(f"Evaluating {feature_type} {name}")
            results = evaluate_model(name, model, X_train, y_train, X_test, y_test, results, feature_type)
            print(results)

        # Optimize hyperparameters
        for name, model in models_.items():
            objective = objectives.get(name)
            if objective is not None:
                print(f"Optimizing {feature_type} {name}")
                results = optimize_hyperparameters(name, model, objective, trials=100, results_dataframe=results, feature_type=feature_type, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
                print(results)

    results.to_csv('results_v2.csv', index=False)

    """# With Feature Selection"""

    # empty dataframe to store results with the columns feature_type, model, with_hypertuning, accuracy, sensitivity, specificity, precision, f1, mcc
    results = pd.DataFrame(columns=['feature_type', 'model', 'with_hypertuning', 'best_params', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'mcc', 'index'])
    feature_types = [ 'selected_features_all_best20','selected_features_all_best50','selected_features_all_best100']
    for feature_type in feature_types:

        # Load the training dataset
        data = pd.read_csv(f'{feature_engineered_data_dir}/TR_{feature_type}.csv')

        # Separate features and target
        X = data.drop(columns=['label', 'id'], axis=1)
        y = data['label']

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Evaluate models without hyperparameters tuning
        for name, model in models.items():
            print(f"Evaluating {feature_type} {name}")
            results = evaluate_model(name, model, X_train, y_train, X_test, y_test, results, feature_type)
            print(results)

        # Optimize hyperparameters
        for name, model in models_.items():
            objective = objectives.get(name)
            if objective is not None:
                print(f"Optimizing {feature_type} {name}")
                results = optimize_hyperparameters(name, model, objective, trials=100, results_dataframe=results, feature_type=feature_type, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
                print(results)

    results.to_csv(f'{feature_engineered_data_dir}/results_20&30&50&100.csv', index=False)

    """# Best Model with Full Training Dataset"""

    # Load the results
    results_without_selected_features = pd.read_csv('results_v2.csv')
    results_with_selected_features = pd.read_csv(f'{feature_engineered_data_dir}/results_20&30&50&100.csv')

    feature_types = ['AAC', 'APAAC', 'CTD', 'DPC', 'PAAC']
    selected_feature_types = ['selected_features_all_best20','selected_features_all_best50','selected_features_all_best100']

    # Combine the feature types
    feature_types.extend(selected_feature_types)

    test_results = []

    # iterate through each row of results
    for feature_type in feature_types:

        # Check if the feature type is selected features
        if 'selected_features' in feature_type:
            # Load the training dataset
            train_data = pd.read_csv(f'{feature_engineered_data_dir}/TR_{feature_type}.csv')
            test_data = pd.read_csv(f'{feature_engineered_data_dir}/TS_{feature_type}.csv')
            results = results_with_selected_features
        else:
            # Load the training dataset
            train_data = pd.read_csv(f'{data_dir}/TR_{feature_type}.csv')
            test_data = pd.read_csv(f'{data_dir}/TS_{feature_type}.csv')
            results = results_without_selected_features

        # Separate features and target
        X_train = train_data.drop(columns=['label', 'id'], axis=1)
        y_train = train_data['label']

        X_test = test_data.drop(columns=['label', 'id'], axis=1)
        y_test = test_data['label']

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # iterate through each model
        for name, model in models.items():
            # get the row of the model

            
            rows = results[(results['feature_type'] == feature_type) & (results['model'] == name)]

            # iterate through each row
            for index, row in rows.iterrows():

                # check whether the model has hyperparameters
                if row['with_hypertuning'] == True:
                    hyperparameters = ast.literal_eval(row['best_params'])
                    # check the model is SVC
                    if row['model'] == 'SVC':
                        hyperparameters = {k[4:]: v for k, v in hyperparameters.items()}
                        # make key 'c' to 'C'
                        hyperparameters['C'] = hyperparameters.pop('c')
                    # set best hyperparameters
                    model.set_params(**hyperparameters)

                # fit model
                model.fit(X_train, y_train)

                # predict
                y_pred = model.predict(X_test)

                # evaluate using accuracy, sensitivity, specificity, precision, f1, mcc
                accuracy = accuracy_score(y_test, y_pred)
                sensitivity = recall_score(y_test, y_pred)
                specificity = recall_score(y_test, y_pred, pos_label=0)
                precision = precision_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                mcc = matthews_corrcoef(y_test, y_pred)

                # append to test_results
                print('feature_type', feature_type,'model', name, 'accuracy', accuracy, 'sensitivity', sensitivity, 'specificity', specificity, 'precision', precision, 'f1', f1)
                
                test_results.append({'feature_type': feature_type, 'model': name, 'with_hypertuning': row['with_hypertuning'], 'best_params': row['best_params'], 'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision, 'f1': f1, 'mcc': mcc, 'index': row['index']})
        print(f'Feature Type: {feature_type} done!')

    test_results = pd.DataFrame(test_results)
    test_results.to_csv('test_results.csv', index=False)

def WriteTextFile(filename,data):
    # Open the file in write mode ('w')
    with open(filename, 'w') as file:
        # Write content to the file
        file.write(data)

def Predict():
    data_dir="processed_dataset"
    models = {
        'SVC': SVC(),
        'XGBClassifier': XGBClassifier(),
        'LGBMClassifier': LGBMClassifier()
    }

    # Load the results
    results = pd.read_csv('output/test_results.csv')
    name=  "XGBClassifier" #results['model'][results['accuracy'].idxmax()]
    model=models[name]
    feature_type = "PAAC" #results['feature_type'][results['accuracy'].idxmax()]
    # feature_types = ['AAC', 'APAAC', 'CTD', 'DPC', 'PAAC']
    # selected_feature_types = ['selected_features_all_best20','selected_features_all_best50','selected_features_all_best100']

    # Combine the feature types
    # feature_types.extend(selected_feature_types)

    test_results = []
    # iterate through each row of results
    # for feature_type in feature_types:

    # Check if the feature type is selected features
    if 'selected_features' in feature_type:
        # Load the training dataset
        train_data = pd.read_csv(f'output/TR_{feature_type}.csv')
        test_data = pd.read_csv(f'output/TS_{feature_type}.csv')
    else:
        # Load the training dataset
        train_data = pd.read_csv(f'{data_dir}/TR_{feature_type}.csv')
        test_data = pd.read_csv(f'{data_dir}/TS_{feature_type}.csv')

    # Separate features and target
    X_train = train_data.drop(columns=['label', 'id'], axis=1)
    y_train = train_data['label']

    X_test = test_data.drop(columns=['label', 'id'], axis=1)
    y_test = test_data['label']

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # iterate through each model
    # get the row of the model

    
    rows = results[(results['feature_type'] == feature_type) & (results['model'] == name)]

    # iterate through each row
    for index, row in rows.iterrows():

        # check whether the model has hyperparameters
        if row['with_hypertuning'] == True:
            hyperparameters = ast.literal_eval(row['best_params'])
            # check the model is SVC
            if row['model'] == 'SVC':
                hyperparameters = {k[4:]: v for k, v in hyperparameters.items()}
                # make key 'c' to 'C'
                hyperparameters['C'] = hyperparameters.pop('c')
            # set best hyperparameters
            model.set_params(**hyperparameters)
        else:
            continue

        # fit model
        model.fit(X_train, y_train)

        # predict
        y_pred = model.predict(X_test)

        # evaluate using accuracy, sensitivity, specificity, precision, f1, mcc
        accuracy = accuracy_score(y_test, y_pred)
        sensitivity = recall_score(y_test, y_pred)
        specificity = recall_score(y_test, y_pred, pos_label=0)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        # append to test_results
        # print( feature_type, name," "*(30-len(feature_type)+len()), accuracy,  sensitivity,  specificity, precision, f1)
        
        test_results.append({'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision, 'f1': f1})
    test_results_pd = pd.DataFrame(test_results)
    print(test_results_pd)
    test_data["predict"]=y_pred
    df_positive = test_data[test_data['id'].str.contains('Positive')]["predict"].to_string(index=False)
    df_negative  = test_data[test_data['id'].str.contains('Negative')]["predict"].to_string(index=False)
    
    WriteTextFile("predictions_pos.txt",df_positive)
    WriteTextFile("predictions_neg.txt",df_negative)
    # df_positive=pd.DataFrame(df_positive)
    # df_negative =pd.DataFrame(df_negative )

import sys

def main():

    if len(sys.argv) != 5:
        print("Error: 4 inputs required.")
        print("Usage: python prediction.py <positive training data> <negative training data> <positive testing data> <negative testing data>")
        sys.exit(1)

    # Get the inputs from the command line
    positive_training_data = sys.argv[1]
    negative_training_data = sys.argv[2]
    positive_testing_data = sys.argv[3]
    negative_testing_data = sys.argv[4]
    # positive_training_data = sys.argv[1] if len(sys.argv) > 1 else "dataset/TR_pos_SPIDER.txt"
    # negative_training_data = sys.argv[2] if len(sys.argv) > 2 else "dataset/TR_neg_SPIDER.txt"
    # positive_testing_data = sys.argv[3] if len(sys.argv) > 3 else "dataset/TS_pos_SPIDER.txt"
    # negative_testing_data = sys.argv[4] if len(sys.argv) > 4 else "dataset/TS_neg_SPIDER.txt"
   
    CreateDataset(positive_training_data, negative_training_data, positive_testing_data, negative_testing_data)
    CreateModels()
    Predict()

if __name__ == "__main__":
    # import os
    # os.system("pip install -r requirements.txt")
    main()
