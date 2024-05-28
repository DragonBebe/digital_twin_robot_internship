import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from sklearn.metrics import max_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from pathlib import Path
from pandas.plotting import scatter_matrix
import logging
import sys
import os
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm


# We provide some supporting function for training a data-driven digital twin for predicting the temperature of motors.


class FaultDetect_OCSVM():
    ''' ### Description
    This is the class for fault detection based on One Class SVM models.

    ### Initialize
    Initialize the class with the following parameters:    
    - ocsvm_mdl: The One Class SVM model.
    - pre_trained: If the provided ocsvm_mdl is pretrained. Default is True.
    - threshold: threshold for the residual error. If the residual error is larger than the threshold, we consider it as a fault. Default is 1.
    - window_size: Size of the sliding window. The previous window size points will be used to create a new feature.
    - sample_step: We take every sample_step points from the window_size. default is 1.

    ### Attributes
    - ocsvm_mdl: The One Class SVM model.
    - pre_trained: If the provided ocsvm_mdl is pretrained. Default is True.
    - threshold: threshold for the residual error. If the residual error is larger than the threshold, we consider it as a fault. Default is 1.
    - window_size: Size of the sliding window. The previous window size points will be used to create a new feature.
    - sample_step: We take every sample_step points from the window_size. default is 1.
    - residual_norm: The stored residual errors for all the normal samples in the current sequence.
      It is used to calculate the threshold adaptively.
    - threshold_int: The threshold value set by the initialization function.

    ### Methods
    - fit(): This method learns the One Class SVM model from the training data. If self.pre_trained is True, it will directly use the pre-trained model.
    - predict(): This method predicts the labels for the input data.
    - run_cross_val(): This method defines a cross validation scheme to test the performance of the model.

    '''

    def __init__(self, ocsvm_mdl, pre_trained: bool = True, threshold: int = 1, window_size: int = 1,
                 sample_step: int = 1, pred_lead_time: int = 1):
        ''' ### Description
        Initialization function.
        '''
        self.ocsvm_mdl = ocsvm_mdl
        self.window_size = window_size
        self.sample_step = sample_step
        self.pred_lead_time = pred_lead_time
        self.threshold = threshold
        self.threshold_int = threshold
        self.pre_trained = pre_trained
        self.residual_norm = []

    def fit(self, df_x, y_label, y_response):
        ''' ### Description
        Learn the regression model from the training data, and use the trained model to predict the labels and the repsonse variables for the training data.
        If self.pre_trained is True, it will directly use the pre-trained model.

        ### Parameters
        - df_x: The training features.
        - y_label: The labels in the training dataset.
        - y_response: The response variable in the training dataset.

        ### Returns
        - y_label_pred: The predicted labels using the best regression model learnt from training data.
        - y_response_pred: The predicted response variable using the best regression model learnt from training data.
        '''
        # Train the regression model if not pretrained.
        if not self.pre_trained:
            # Train a regression model first, based on the normal samples.
            # Align indices
            df_x_normal = copy.deepcopy(df_x)
            # Initialize a counter for numbering
            counter = 0
            flag = False
            # Iterate over each row in df
            for i in range(len(y_label)):
                value = y_label.iloc[i]
                if i > 0:
                    if value == 0 and y_label.iloc[i - 1] == 1:
                        flag = True
                        counter += 1
                    if value == 1 and y_label.iloc[i - 1] == 0:
                        flag = False
                    if flag:
                        df_x_normal.at[df_x_normal.index[i], 'test_condition'] += f'_{counter}'

            df_x_normal = df_x_normal[y_label == 0]
            y_response_normal = y_response[y_label == 0]

            # Train the regression model.
            x_tr, y_temp_tr = prepare_sliding_window(df_x=df_x_normal, y=y_response_normal,
                                                     window_size=self.window_size, sample_step=self.sample_step,
                                                     prediction_lead_time=self.pred_lead_time, mdl_type='reg')
            self.reg_mdl = self.reg_mdl.fit(x_tr, y_temp_tr)

        # Calculate and return the predicted labels and response variable for the training data.
        y_label_pred, y_response_pred = self.predict(df_x, y_response)

        return y_label_pred, y_response_pred

    def predict(self, df_x_test, y_response_test):
        ''' ### Description
        Predict the labels using the trained regression model and the measured response variable.
        Note that if a fault is predicted, the predicted, not measured response variable will be used to concatenate features
        for predicting the values of next response variable.

        ### Parameters
        - df_x_test: The testing features.
        - y_response_test: The measured response variable.

        ### Return
        - y_label_pred: The predicted labels.
        - y_response_pred: The predicted response variable.
        '''
        # Get parameters.
        window_size = self.window_size
        sample_step = self.sample_step
        prediction_lead_time = self.pred_lead_time

        # Get the sequence names.
        sequence_name_list = df_x_test['test_condition'].unique().tolist()

        # Initial values
        y_label_pred = []
        y_response_pred = []

        # Process sequence by sequence.
        for name in tqdm(sequence_name_list):
            # Reset the stored residual errors for normal samples and the threshold value.
            self.residual_norm = []
            self.threshold = self.threshold_int

            # Extract one sequence.
            df_x_test_seq = df_x_test[df_x_test['test_condition'] == name]
            y_temp_test_seq = y_response_test[df_x_test['test_condition'] == name]
            y_temp_local = copy.deepcopy(y_temp_test_seq)

            # Initial values of the prediction.
            # Length is len - window_size + 1 because we need to use the sliding window to define features.
            y_label_pred_tmp = np.zeros(len(df_x_test_seq) - window_size + 1)  # Predicted label.
            y_temp_pred_tmp = np.zeros(len(df_x_test_seq) - window_size + 1)  # Predicted temperature.

            # Making the prediction using a sequential approach.
            for i in range(window_size, len(df_x_test_seq) + 1):
                # Get the data up to current moment i-1.
                tmp_df_x = df_x_test_seq.iloc[i - window_size:i, :]
                tmp_y_temp_measure = y_temp_local.iloc[i - window_size:i]

                # Use the same sliding window to generate features.
                feature_x, _ = concatenate_features(df_input=tmp_df_x, y_input=tmp_y_temp_measure, X_window=[],
                                                    y_window=[],
                                                    window_size=window_size, sample_step=sample_step,
                                                    prediction_lead_time=prediction_lead_time, mdl_type='reg')

                # Make prediction.
                tmp_y_label_pred, tmp_y_temp_pred, tmp_residual = self.predict_label_by_reg_base(X=feature_x,
                                                                                                 y_temp_measure=
                                                                                                 tmp_y_temp_measure.iloc[
                                                                                                     -1])

                # Save the prediction at the current moment i.
                y_label_pred_tmp[i - window_size] = tmp_y_label_pred[-1]
                y_temp_pred_tmp[i - window_size] = tmp_y_temp_pred[-1]

                # If we predict a failure, we replace the measure with the predicted temperature.
                # This is to avoid accumulation of errors.
                if tmp_y_label_pred[-1] == 1 and tmp_residual <= self.abnormal_limit:
                    y_temp_local.iloc[i - 1] = tmp_y_temp_pred[-1]

            # Save the results and proceed to the next sequence.
            y_label_pred.extend(y_label_pred_tmp)
            y_response_pred.extend(y_temp_pred_tmp)

        return y_label_pred, y_response_pred

    def run_cross_val(self, df_x, y_label, n_fold=5, single_run_result=True):
        ''' ## Description
        Run a k-fold cross validation based on the testing conditions. Each test sequence is considered as a elementary part in the data.

        ## Parameters:
        - df_x: The dataframe containing the features. Must have a column named "test_condition".
        - y_label: The target variable, i.e., failure.
        - n_fold: The number of folds. Default is 5.
        - single_run_result: Whether to return the single run result. Default is True.

        ## Return
        - perf: A dataframe containing the performance indicators.
        '''

        # Get the unique test conditions.
        test_conditions = df_x['test_condition'].unique().tolist()

        # Define the cross validator.
        kf = KFold(n_splits=n_fold)

        # Set initial values for perf to store the performance of each run.
        perf = np.zeros((n_fold, 4))

        counter = 0
        for train_index, test_index in kf.split(test_conditions):
            # Get the dataset names.
            names_train = [test_conditions[i] for i in train_index]
            names_test = [test_conditions[i] for i in test_index]

            # If not pretrained, train the model.
            if not self.pre_trained:
                df_tr = df_x[df_x['test_condition'].isin(names_train)]
                self.ocsvm_mdl = self.ocsvm_mdl.fit(df_tr)

            # Extract the training and testing data.
            df_test = df_x[df_x['test_condition'].isin(names_test)]
            y_test = y_label[df_x['test_condition'].isin(names_test)]

            # Predict for the testing data.
            y_pred = self.predict(df_test)

            # Calculate the performance.
            accuracy, precision, recall, f1 = cal_classification_perf(y_test, pd.Series(y_pred))
            perf[counter, :] = np.array([accuracy, precision, recall, f1])

            # Show single run results.
            if single_run_result:
                fig_1 = plt.figure(figsize=(8, 12))
                ax_test_true = fig_1.add_subplot(2, 1, 1)
                ax_test_pred = fig_1.add_subplot(2, 1, 2)
                show_clf_result_single_run(y_test, y_pred, ax_test_true, ax_test_pred, suffix='testing')

                plt.show()

            counter += 1

        return pd.DataFrame(data=perf, columns=['Accuracy', 'Precision', 'Recall', 'F1 score'])


def extract_selected_feature(df_data: pd.DataFrame, feature_list: list, motor_idx: int, mdl_type: str):
    ''' ### Description
    Extract the selected features and the response variable from the dataframe.

    ### Parameters
    df_data: The dataframe containing the data.
    feature_list: The list of features to be used.
    motor_idx: The index of the motor.
    mdl_type: The type of the model. 'clf' for classification, 'reg' for regression.

    ### Return
    df_x: The dataframe containing the features.
    y: The response variable.
    '''

    # Create a copy of feature_list
    feature_list_local = copy.deepcopy(feature_list)
    # Get the name of the response variable.
    if mdl_type == 'clf':
        y_name = f'data_motor_{motor_idx}_label'
    elif mdl_type == 'reg':
        y_name = f'data_motor_{motor_idx}_temperature'
    else:
        raise ValueError('mdl_type must be \'clf\' or \'reg\'.')

    # Remove the y from the feature
    if y_name in feature_list_local:
        feature_list_local.remove(y_name)

    # Seperate features and the response variable.
    # Remove the irrelavent features.
    feature_list_local.append('test_condition')
    df_x = df_data[feature_list_local]
    # Get y.
    y = df_data.loc[:, y_name]

    return df_x, y


def run_cv_one_motor(motor_idx, df_data, mdl, feature_list, n_fold=5, window_size=1, sample_step=1,
                     single_run_result=True, mdl_type='clf'):
    ''' ### Description
    Run cross validation for a given motor and return the performance metrics for each cv run.
    Can be used for both classification and regression models.

    ### Parameters
    - motor_idx: The index of the motor.
    - df_data: The dataframe containing the data. Must contain a column named 'test_condition'.
    - mdl: The model to be trained. Must have a fit() and predict() method.
    - feature_list: The list of features to be used for the model.
    - n_fold: The number of folds for cross validation. Default is 5. The training and testing data are split by sequence.
    So one needs to make sure n_fold <= the number of sequences.
    - window_size: The window size for the sliding window. Default is 0, which means no sliding window.
    - sample_step: We take every sample_step points from the window_size. default is 1.
    - single_run_result: Whether to return the performance metrics for each cv run. Default is True.
    - mdl_type: The type of the model. Can be 'clf' or 'reg'. Default is 'clf'.

    ### Return
    - df_perf: The dataframe containing the performance metrics for each cv run.
    If mdl_type is 'clf', the performance metrics are accuracy, precision, recall, and f1 score.
    If mdl_type is 'reg', the performance metrics are max error, mean squared error, and out-of-threshold percentage.

    '''
    # Extract the selected features.
    df_x, y = extract_selected_feature(df_data, feature_list, motor_idx, mdl_type)

    print(f'Model for motor {motor_idx}:')
    # Run cross validation.
    df_perf = run_cross_val(mdl, df_x, y, n_fold=n_fold, threshold=threshold, window_size=window_size,
                            sample_step=sample_step, prediction_lead_time=prediction_lead_time,
                            single_run_result=single_run_result, mdl_type=mdl_type)
    print(df_perf)
    print('\n')

    # Print the mean performance and standard error.
    print('Mean performance metric and standard error:')
    for name, metric, error in zip(df_perf.columns, df_perf.mean(), df_perf.std()):
        print(f'{name}: {metric:.4f} +- {error:.4f}')
    print('\n')

    return df_perf


def cal_classification_perf(y_true, y_pred):
    ''' ### Description
    This function calculates the classification performance: Accuracy, Precision, Recall and F1 score.
    It considers different scenarios when divide by zero could occur for Precision, Recall and F1 score calculation.

    ### Parameters:
    - y_true: The true labels.
    - y_pred: The predicted labels.

    ### Return:
    - accuracy: The accuracy.
    - precision: The precision.
    - recall: The recall.
    - f1: The F1 score.
    '''
    accuracy = accuracy_score(y_true, y_pred)
    # Only when y_pred contains no zeros, and y_true contains no zeros, set precision to be 1 when divide by zero occurs.
    if sum(y_true) == 0 and sum(y_pred) == 0:
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        f1 = f1_score(y_true, y_pred, zero_division=1)
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

    return accuracy, precision, recall, f1


def show_clf_result(y_tr, y_test, y_pred_tr, y_pred):
    ''' ## Description
    This subfunction visualize the performance of the fitted model on both the training and testing dataset for the classfication model.

    ## Parameters
    - y_tr: The training labels.
    - y_test: The testing labels.
    - y_pred_tr: The predicted labels on the training dataset.
    - y_pred: The predicted labels on the testing dataset.
    '''
    fig_1 = plt.figure(figsize=(16, 12))
    ax_tr_true = fig_1.add_subplot(2, 2, 1)
    ax_tr_pred = fig_1.add_subplot(2, 2, 3)
    ax_test_true = fig_1.add_subplot(2, 2, 2)
    ax_test_pred = fig_1.add_subplot(2, 2, 4)

    show_clf_result_single_run(y_true=y_tr, y_pred=y_pred_tr, ax_tr=ax_tr_true, ax_pred=ax_tr_pred, suffix='training')
    show_clf_result_single_run(y_true=y_test, y_pred=y_pred, ax_tr=ax_test_true, ax_pred=ax_test_pred, suffix='testing')

    plt.show()


# Subfunction for create the sliding window.
def concatenate_features(df_input, y_input, X_window, y_window, window_size=1, sample_step=1, prediction_lead_time=1,
                         mdl_type='clf'):
    ''' ### Description
    This function takes every sample_step point from a interval window_size, and concatenate the extracted
    features into a new feature list X_window. It extracts the corresponding y in y_window.

    ### Parameters
    - df_input: The original feature matrix.
    - y_input: The original response variable.
    - X_window: A list containing the existing concatenated features. Each element is a list of the new features.
    - y_window: A list containing the existing concatenated response variable.
    - window_size: Size of the sliding window. The points in the sliding window will be used to create a new feature.
    - sample_step: We take every sample_step points from the window_size. default is 1.
    - prediction_lead_time: We predict y at t using the previous measurement of y up to t-prediction_lead_time. Default is 1.
    - mdl_type: The type of the model. 'clf' for classification, 'reg' for regression. Default is 'clf'.

    ### Return
    - X_window: The X_window after adding the concatenated features from df_input.
    - y_window: The y_window after adding the corresponding y.
    '''

    # Get the index of the last element in the dataframe.
    idx_last_element = len(df_input) - 1
    # Get the indexes the sampled feature in the window, from the last element of the dataframe.
    idx_samples = list(reversed(range(idx_last_element, idx_last_element - window_size, -1 * sample_step)))
    # Get the sample X features, and concatenate
    new_features = df_input.iloc[idx_samples].drop(columns=['test_condition']).values.flatten().tolist()

    # If mdl_type is 'reg', we need to add the past ys to the new_features.
    if mdl_type == 'reg':
        if prediction_lead_time <= 1:  # It is meaningless to add the current y as it is what we need to predict. So the prediction_leatime >= 1.
            prediction_lead_time = 1
        if prediction_lead_time < window_size and window_size > 1:  # Otherwise no need to add y_prev as they are beyond the window_size.
            tmp_idx_pred = [x for x in idx_samples if x <= idx_last_element - prediction_lead_time]
            new_features.extend(y_input.iloc[tmp_idx_pred].values.flatten().tolist())

    # Add the added features and the corresponding ys into X_window and y_window.
    X_window.append(new_features)  # New features
    y_window.append(y_input.iloc[idx_last_element])  # Corresponding y

    return X_window, y_window

# Sliding the window to create features and response variables.
def prepare_sliding_window(df_x, y, sequence_name_list=None, window_size=1, sample_step=1, prediction_lead_time=1, mdl_type='clf'):
    ''' ## Description
    Create a new feature matrix X and corresponding y, by sliding a window of size window_size.

    ## Parameters:
    - df_x: The dataframe containing the features. Must have a column named "test_condition".
    - y: The target variable.
    - sequence_name_list: The list of sequence names, each name represents one sequence.
    - window_size: Size of the sliding window. The points in the sliding window will be used to create a new feature.
    - sample_step: We take every sample_step points from the window_size. default is 1.
    - prediction_lead_time: We predict y at t using the previous measurement of y up to t-prediction_lead_time. Default is 1.
    - mdl_type: The type of the model. 'clf' for classification, 'reg' for regression. Default is 'clf'.

    ## Return
    - X: Dataframe of the new features.
    - y: Series of the response variable.
    '''
    X_window = []
    y_window = []

    # If no sequence_list is given, extract all the unique values from 'test_condition'.
    if sequence_name_list is None:
        sequence_name_list = df_x['test_condition'].unique().tolist()

    # Ensure y is a DataFrame for consistent indexing
    y = pd.Series(y, index=df_x.index)  # Ensure y has the same index as df_x

    # Process sequence by sequence.
    for name in sequence_name_list:
        # Extract one sequence.
        df_tmp = df_x[df_x['test_condition'] == name].reset_index(drop=True)
        y_tmp = y.loc[df_tmp.index].reset_index(drop=True)  # Use .loc to ensure index alignment

        # Do a loop to concatenate features by sliding the window.
        for i in range(window_size, len(df_tmp) + 1):
            X_window, y_window = concatenate_features(df_input=df_tmp.iloc[i - window_size:i, :], y_input=y_tmp.iloc[i - window_size:i],
                                                      X_window=X_window, y_window=y_window, window_size=window_size, sample_step=sample_step, prediction_lead_time=prediction_lead_time, mdl_type=mdl_type)

    # Transform into dataframe.
    X_window = pd.DataFrame(X_window)
    y_window = pd.Series(y_window)

    return X_window, y_window

def read_all_csvs_one_test(folder_path: str, test_id: str = 'unknown', pre_processing: callable = None) -> pd.DataFrame:
    ''' ## Description
    Combine the six CSV files (each for a motor) in a folder into a single DataFrame. The test condition in the input will be recorded as a column in the combined dataframe.

    ## Parameters
    - folder_path: Path to the folder containing the six CSV files
    - test_condition: The condition of the test. Should be read from "Test conditions.xlsx". Default is 'unknown'.
    - pre_processing: A function handle to the data preprocessing function. Default is None.

    ## Return
    - combined_df: A DataFrame containing all the data from the CSV files.
    '''

    # Get a list of all CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # Create an empty DataFrame to store the combined data
    combined_df = pd.DataFrame()

    # Iterate over the CSV files in the folder
    for file in csv_files:
        # Construct the full path to each CSV file
        file_path = os.path.join(folder_path, file)

        # Read each CSV file into a DataFrame
        df = pd.read_csv(file_path)
        # Drop the time. Will add later.
        df = df.drop(labels=df.columns[0], axis=1)

        # Apply the pre-processing.
        if pre_processing:
            pre_processing(df)

        # Extract the file name (excluding the extension) to use as a prefix
        file_name = os.path.splitext(file)[0]

        # Add a prefix to each column based on the file name
        df = df.add_prefix(f'{file_name}_')

        # Concatenate the current DataFrame with the combined DataFrame
        combined_df = pd.concat([combined_df, df], axis=1)

    # Add time and test condition
    df = pd.read_csv(file_path)
    combined_df = pd.concat([df['time'], combined_df], axis=1)

    # Calculate the time difference since the first row
    time_since_first_row = combined_df['time'] - combined_df['time'].iloc[0]
    # Replace the 'time' column with the time difference
    combined_df['time'] = time_since_first_row

    combined_df.loc[:, 'test_condition'] = test_id

    # Drop the NaN values, which represents the first n data points in the original dataframe.
    combined_df.dropna(inplace=True)

    return combined_df

def read_all_test_data_from_path(base_dictionary: str, pre_processing: callable = None, is_plot=True) -> pd.DataFrame:
    ''' ## Description
    Read all the test data from a folder. The folder should contain subfolders for each test. Each subfolder should contain the six CSV files for each motor.
    The test condition in the input will be recorded as a column in the combined dataframe.

    ## Parameters
    - base_dictionary: Path to the folder containing the subfolders for each test.
    - pre_processing: A function handle to the data preprocessing function.Default is None.
    - is_plot: Whether to plot the data. Default is True.

    ## Return
    - df_data: A DataFrame containing all the data from the CSV files.
    '''

    # Get all the folders in the base_dictionary
    path_list = os.listdir(base_dictionary)
    # Only keep the folders, not the excel file.
    path_list_sorted = sorted(path_list)
    path_list = path_list_sorted[:-1]

    # Read the data.
    df_data = pd.DataFrame()
    for tmp_path in path_list:
        path = base_dictionary + tmp_path
        tmp_df = read_all_csvs_one_test(path, tmp_path, pre_processing)
        df_data = pd.concat([df_data, tmp_df])
        df_data = df_data.reset_index(drop=True)

    # Read the test conditions
    df_test_conditions = pd.read_excel(base_dictionary + 'Test conditions.xlsx')

    # Visulize the data
    if is_plot:
        for selected_sequence_idx in path_list:
            filtered_df = df_data[df_data['test_condition'] == selected_sequence_idx]

            print('{}: {}\n'.format(selected_sequence_idx,
                                    df_test_conditions[df_test_conditions['Test id'] == selected_sequence_idx][
                                        'Description']))

            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
            for ax, col in zip(axes.flat, ['data_motor_1_position', 'data_motor_2_position', 'data_motor_3_position',
                                           'data_motor_1_temperature', 'data_motor_2_temperature',
                                           'data_motor_3_temperature',
                                           'data_motor_1_voltage', 'data_motor_2_voltage', 'data_motor_3_voltage']):
                label_name = col[:13] + 'label'
                tmp = filtered_df[filtered_df[label_name] == 0]
                ax.plot(tmp['time'], tmp[col], marker='o', linestyle='None', label=col)
                tmp = filtered_df[filtered_df[label_name] == 1]
                ax.plot(tmp['time'], tmp[col], marker='x', color='red', linestyle='None', label=col)
                ax.set_ylabel(col)

            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
            for ax, col in zip(axes.flat, ['data_motor_4_position', 'data_motor_5_position', 'data_motor_6_position',
                                           'data_motor_4_temperature', 'data_motor_5_temperature',
                                           'data_motor_6_temperature',
                                           'data_motor_4_voltage', 'data_motor_5_voltage', 'data_motor_6_voltage']):
                label_name = col[:13] + 'label'
                tmp = filtered_df[filtered_df[label_name] == 0]
                ax.plot(tmp['time'], tmp[col], marker='o', linestyle='None', label=col)
                tmp = filtered_df[filtered_df[label_name] == 1]
                ax.plot(tmp['time'], tmp[col], marker='x', color='red', linestyle='None', label=col)
                ax.set_ylabel(col)

            plt.show()

    return df_data

def show_clf_result_single_run(y_true, y_pred, ax_tr, ax_pred, suffix='training'):
    ''' ### Description
    This function plot the predictin results for a classifier, and print the performance metrics.
    '''
    # Plots
    ax_tr.set_xlabel('index of data point', fontsize=15)
    ax_tr.set_ylabel('y', fontsize=15)
    ax_tr.set_title(f'{suffix}: Truth', fontsize=20)
    ax_tr.plot(range(len(y_true)), y_true, 'xb', label='Truth')
    ax_tr.legend()

    ax_pred.set_xlabel('index of data points', fontsize=15)
    ax_pred.set_ylabel('y', fontsize=15)
    ax_pred.set_title(f'{suffix}: Prediction', fontsize=20)
    ax_pred.plot(range(len(y_pred)), y_pred, 'xb', label='Truth')
    ax_pred.legend()

    # Performance indicators
    # Show the model fitting performance.
    acc, pre, recall, f1 = cal_classification_perf(y_true, y_pred)
    print('\n New run:\n')
    print(f'{suffix} performance, accuracy is: ' + str(acc))
    print(f'{suffix} performance, precision is: ' + str(pre))
    print(f'{suffix} performance, recall: ' + str(recall))
    print(f'{suffix} performance, F1: ' + str(f1))
    print('\n')



