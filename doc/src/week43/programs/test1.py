import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier # Example estimator

# Assuming y_true and y_pred are your true and predicted labels
# and you have a trained estimator (e.g., knn)
# Example data:
y_true = [0, 1, 0, 1, 0, 1, 0, 1]
y_pred = [0, 0, 0, 1, 1, 1, 0, 0]
classes = [0, 1]

# Calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=classes)

# Create and plot the display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues) # Customize colormap if desired
plt.title("Confusion Matrix")
plt.show()
