# CPSC 473 - Introduction to Data Mining
# Team Project - Audio Analysis and Classification
# .wav Classifier (main)

# Internal Imports
from KNN import knn_classify
from LogisticRegression import log_reg_classify
from PreProcessing import *

# External Imports
from csv import reader as csv_reader
from os import listdir
from pandas import read_csv, concat
from random import randint
from time import time

def main():
    print("CPSC 473 Audio Analysis and Classification Project - .wav Classifier" + '\n')

    # Setup all required directory/file references
    wav_dirs = ['.\\wavs\\' + wav_dir for wav_dir in listdir('.\\wavs')]
    wav_data_files = dict()
    for wav_dir in wav_dirs:
        wav_data_files[wav_dir] = '.\\wavData\\' + wav_dir[7:] + 'Data.txt'
    expected_results_file = "ExpectedResults.txt"
    knn_output_file = "KNNOutput.txt"
    log_reg_output_file = "LogRegOutput.txt"

    # Prompt user regarding need for .wav file preprocessing
    preprocess_wavs = ''
    while preprocess_wavs not in ['Y', 'N']:
        preprocess_wavs = input("Would you like to preprocess the .wav files?\n" + 
        "  Y: Yes\n" + 
        "  N: No (Preprocessing output must already exist in .\\wavData\\ from a previous run of the program)\n").upper()

    # Prompt user regarding choice of training/testing data designation
    test_fold_number = None
    while test_fold_number not in range(0, 11):
        test_fold_number = int(input("Which fold of .wav files would you like to designate as test data?\n" + 
        "  0: Random\n" + 
        "  1 ... 10: Fold 1 ... 10\n"))
    if test_fold_number == 0:
        test_fold_number = randint(1, 10)
    print("Fold selected:", test_fold_number)

    # Prompt user regarding choice of .wav file preprocessing methodology and preprocess .wav files accordingly
    if preprocess_wavs == 'Y':
        preprocess_method = ''
        while preprocess_method not in range(0, 3):
            preprocess_method = int(input("How would you like the .wav files to be preprocessed?\n" + 
            "  0: Via earlier preprocessing methodology - Various attributes based on intuitive knowledge of audio, such as average amplitudes within distinct frequency bands\n" + 
            "  1: Via later preprocessing methodology - Averages of Mel-frequency Cepstral Coefficients (MFCCs), which (approximately) define timbre\n" +
            "  2: A combination of selections 1 and 2\n"))
        
            start = time()
            print("Preprocessing .wav files...")

            for wav_dir in wav_dirs:
                print("Preprocessing .wav files in", wav_dir)
                begin_preprocessing(wav_dir, wav_data_files[wav_dir], preprocess_method)

            preprocess_time = time() - start
            print("Preprocessed .wav files in all folds in", str(int(round(preprocess_time, 0))) + "s")
    
    # Distinguish training/testing data as per user choice of training/testing data designation
    training_data_files = []
    testing_data_file = None

    for wav_dir in wav_data_files:
        wav_data_file = wav_data_files[wav_dir]

        fold_number = ''.join(char for char in wav_data_file if char.isnumeric())
        if fold_number == str(test_fold_number):
            # Write expected classifications to .\ExpectedResults.txt
            testing_data_file = wav_data_file
            test_fold_data = read_csv(testing_data_file)
            test_fold_classes = test_fold_data["class"].values
            with open(expected_results_file, "w+") as output_file:
                for test_fold_class in test_fold_classes:
                    output_file.write(str(test_fold_class) + '\n')
        else:
            training_data_files.append(wav_data_file)

    # Compile training and testing data for passing to KNN and Logistic Regression algorithms
    training_data = concat([read_csv(training_data_file) for training_data_file in training_data_files])
    testing_data = read_csv(testing_data_file, header = 0).drop("class", axis = 1).values

    # Build a KNN model using Training Data and classify Testing Data via said model
    print("Classifying Test Data via KNN...")
    start = time()
    knn_classify(training_data, testing_data, knn_output_file)
    knn_accuracy = check_classifications(expected_results_file, knn_output_file)
    knn_time = time() - start

    # Build a Logistic Regression model using Training Data and classify Testing Data via said model
    print("Classifying Test Data via Logistic Regression...")
    start = time()
    log_reg_classify(training_data, testing_data, log_reg_output_file)
    log_reg_accuracy = check_classifications(expected_results_file, log_reg_output_file)
    log_reg_time = time() - start

    # Print average classification accuracy results to console
    print("KNN classifications are, on average,", str(knn_accuracy), '%', "accurate")
    print("KNN training and classification completed within", str(round(knn_time)), "s")
    print("See", knn_output_file, "for the full KNN output")

    print("Logistic Regression classifications are, on average,", str(log_reg_accuracy), '%', "accurate")
    print("Logistic Regression training and classification completed within", str(round(log_reg_time)), "s")
    print("See", log_reg_output_file, "for the full Logistic Regression output")

####

def begin_preprocessing(dir, output_file, preprocess_method, training_data = True):
    """Begins processing the wavs in the given directory through the audio preprocess_method and outputs the desired features

        Parameters
        ----------
        dir : str
            Directory the wavs can be found in
        output_file : str
            File name to output data to
        preprocess_method : int
            Preprocessing methodology to employ
        training_data : boolean
            Boolean that handles whether the method is processing training data or testing/unknown data (no longer in use; now defaults to True)
    """

    with open(output_file, "w+") as output_file:
        files_w_attr = []
        file_names = listdir(dir)
        amount_files = len(file_names)

        # Handled current processing percentage counting
        print("0%", end="...")
        counter = 0
        current_percent = 0
        # For all the file names in the directory
        for file_name in file_names:
            counter += 1
            # Calculate current percentage
            current_percent = percent_finished(counter, amount_files, current_percent)
            # Don't include not_a_virus.exe files
            if file_name.endswith(".wav"):
                # Gets some information about the file for librosa processing
                file = get_wav_file(dir, file_name)
                # Get useful attributes for classification for current file
                file_w_attr = get_file_w_attr(file, preprocess_method)
                files_w_attr.append(file_w_attr)

                # Append necessary column names to first line of CSV file
                if counter == 1:
                    for key in file_w_attr:
                        if key != list(file_w_attr.keys())[-1]:
                            if key == "name" and training_data == 1:
                                output_file.write("class,")
                            elif key != "name" and key != "fp_values" and key != "sample_rate" and key != "dur":
                                output_file.write(str(key) + ",")
                        else:
                            output_file.write(str(key) + "\n")

                # If we hit 1000 files processed, dump data to file to ease RAM usage
                if len(files_w_attr) > 1000:
                    for file_w_attr_dump in files_w_attr:
                        output_to_file(file_w_attr_dump, output_file, training_data)
                    files_w_attr = []

        # Dump the rest of the data to the file
        for file_w_attr_dump in files_w_attr:
            output_to_file(file_w_attr_dump, output_file, training_data)

def output_to_file(file_attrs, output_file, training_data):
    """Outputs the file attribute data to the output file

        Parameters
        ----------
        file_attrs : dict
            Map of the important data points
        output_file : IOWrapper
            Opened file to write to
        training_data : boolean
            Boolean that handles whether it's processing training data or unknown data
    """

    # Iterate through dict
    for key in file_attrs:
        if key == "name":
            # Pass file name to classify_wavs method
            classification = classify_wavs(file_attrs[key])
            if classification != None and training_data == 1:
                output_file.write(str(classification) + ",")
            # None classification means there is background noise (salience = 2), do not use data point
            elif classification == None:
                break
        # These keys are not important to output and can be skipped, others will be output to file
        elif key != "fp_values" and key != "sample_rate" and key != "dur":
            if key != list(file_attrs.keys())[-1]:
                output_file.write(str(file_attrs[key]) + ",")
            else:
                output_file.write(str(file_attrs[key]) + "\n")

def percent_finished(counter, total, current_percent):
    """Prints the percent finished in multiples of 10 if it is a new percentage

        Parameters
        ----------
        counter : int
            numerator
        total : int
            demoninator
        current_percent : int
            last printed percent
    """

    percent_finished = round((counter / total) * 100)
    if percent_finished % 10 == 0 and percent_finished != current_percent:
        if percent_finished != 100:
            print(str(percent_finished), end="%...")
        else:
            print(str(percent_finished), end="%...\n")
    return percent_finished

def classify_wavs(filename):
    """Finds file's classification in Metadata.csv

        Parameters
        ----------
        filename : string
            filename of wav file to find in Metadata.csv
    """

    # Iterate through CSV to find filename, when found, check salience, if it is 1, return the classification, if 2, return None
    csv_file = csv_reader(open("Metadata.csv", 'r'), delimiter=",")
    for row in csv_file:
        if filename == row[0] and row[4] == "1":
            return row[7]
        elif filename == row[0] and row[4] == "2":
            return

def check_classifications(expected_results_file_name, output_file_name):
    """Takes in PreProcessing output files and prints model accuracy

        Parameters
        ----------
        expected_results_file_name : str
            Name of the file with the expected results
        output_file_name : str
            Name of the file with the actual model output
    """

    expected_output = []
    knn_output = []
    with open(expected_results_file_name, "r") as expected_results:
        expected_output = expected_results.read().splitlines()
    with open(output_file_name, "r") as results:
        knn_output = results.read().splitlines()
    
    if(len(expected_output) != len(knn_output)):
        print("Something went wrong... cannot calculate classification accuracy due to file length mismatch.")
        return

    num_results = len(expected_output)
    num_correct = 0

    for expected, actual in zip(expected_output, knn_output):
        if (expected in actual):
            num_correct += 1

    accuracy = round((num_correct / num_results) * 100)
    return accuracy

####

# Driver:
if __name__ == "__main__":
    main()