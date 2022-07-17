# CPSC 473 - Introduction to Data Mining
# Team Project - Audio Analysis and Classification

# External Imports
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler

def knn_classify(training_data, testing_data, output_file_name):
    """Takes in training and testing data (CSVs) and outputs a file with the KNN classifications of the given testing data

        Parameters
        ----------
        training_data : str
            Training data
        testing_file_name : str
            Testing data
        output_file_name : str
            Name of the file the method should output classifications to
    """

    # Separates data into a matrix of independent variables, and a vector of targets
    X = training_data.drop("class", axis = 1)
    X = X.values

    Y = training_data["class"]
    Y = Y.values

    classify = testing_data

    # Standardization/Scaling
    standard_scaler = StandardScaler().fit(X)
    X = standard_scaler.transform(X)
    standard_scaler = StandardScaler().fit(classify)
    classify = standard_scaler.transform(classify)

    with open(output_file_name, "w") as out:
        for new_data_point in classify:
            # Get euclidian distances from current point to all points
            distances = norm(X - new_data_point, axis = 1)
            k = 5
            # Finds IDs of k items closest to the new point
            neighbour_ids = distances.argsort()[:k]

            # Weighted KNN calculations
            weight_factors = dict()
            for neighbour in neighbour_ids:
                key = Y[neighbour]
                weight = (1 / ((distances[neighbour]) ** 2))
                if key in weight_factors:
                    weight_factors[key] += weight
                else:
                    weight_factors[key] = weight

            classification = max(weight_factors, key=weight_factors.get)
            certainty = round(((weight_factors[classification] / sum(weight_factors.values())) * 100))

            # Output result of KNN processing to file with certainty
            out.write(str(classification) + " with " + str(certainty) + "% certainty\n")