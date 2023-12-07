# Import necessary libraries
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', parser='auto') # Ladda MNIST dataset från OpenMl

X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=42) # Dela upp dataset till två olika delar (training och testing)

# Normalize pixel values to the range [0, 1]
X_train /= 255.0
X_test /= 255.0

mlp_classifier = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                               solver='sgd', verbose=10, random_state=1,
                               learning_rate_init=.1) # Skapande av klassifikation

mlp_classifier.fit(X_train, y_train) # Träna upp klassifikation 

y_pred = mlp_classifier.predict(X_test) # Predicitons

accuracy = accuracy_score(y_test, y_pred) # Beräkna noggranhet
print(f"Accuracy: {accuracy * 100:.2f}%")