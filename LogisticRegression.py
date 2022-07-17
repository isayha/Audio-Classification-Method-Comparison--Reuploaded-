# CPSC 473 - Introduction to Data Mining
# Team Project - Audio Analysis and Classification

# External Imports
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def log_reg_classify(training_data, testing_data, output_file_name):
    # Divide dependent variables (classes) and independent variables within the given training data
    dep_vars = training_data["class"].values
    indep_vars = training_data.drop("class", axis = 1).values

    # Standardize the independent variables within the given training data to accommodate for outliers
    standard_scaler = StandardScaler().fit(indep_vars)
    indep_vars_scaled = standard_scaler.transform(indep_vars)

    # Fit a multi-classifying (via One vs. Rest technique) Logistic Regression algorithm to the given training data
    log_reg_model = LogisticRegression(multi_class = 'ovr', n_jobs = -1).fit(indep_vars_scaled, dep_vars)

    # Standardize the indepedent variables within the given testing data as were those within the given training data
    standard_scaler = StandardScaler().fit(testing_data)
    testing_data_scaled = standard_scaler.transform(testing_data)

    # Perform logistic regression and write individual classifications, alongside their confidences (probabilities), to the specified output file
    output_file = open(output_file_name, 'w')
    predictions = log_reg_model.predict(testing_data_scaled)
    prediction_probabilities = log_reg_model.predict_proba(testing_data_scaled)
    for index in range(0, len(predictions)):
        prediction = str(predictions[index])
        prediction_probability = "{:.0%}".format(max(prediction_probabilities[index]))
        output_file.write(prediction + ' with ' + prediction_probability + ' confidence' + '\n')