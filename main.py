from util import *
from gaussian_mixtures import *

def compute_confusion_matrix(actual_labels, predicted_labels, num_classes):
    confusion_mat = np.zeros((num_classes, num_classes), dtype=int)

    for actual, pred in zip(actual_labels, predicted_labels):
        confusion_mat[actual, pred] += 1

    return confusion_mat

def iris_classification():
    train_x, train_y, test_x, test_y = get_iris_data()
    train_x, _, test_x = normalize(train_x, train_x, test_x)
    train_x, test_x = train_x, test_x

    test_y = test_y.flatten()
    train_y = train_y.flatten()
    num_class = (np.unique(train_y)).shape[0]

    # X = np.concatenate([train_x, test_x])

    # Initialize and fit Gaussian Mixture Model
    gmm = GaussianMixtureModel(num_components=num_class)
    gmm.fit(train_x)

    # Predictions
    test_predictions = gmm.predict(test_x)

    print("actual_labels:", test_y)
    print("predicted_labels:", test_predictions)

    # Compute confusion matrix
    confusion_matrix = compute_confusion_matrix(test_y, test_predictions, num_class)
    print("Confusion Matrix:")
    print(confusion_matrix)

if __name__ == "__main__":
    iris_classification()
