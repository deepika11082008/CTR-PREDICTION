# Predictive Modeling for Click-Through Rate Optimization
# By ConnectSphere Digital

import pandas as pd
import numpy as np
import matplotlib
# Use a non-interactive backend so script can run in headless/CI environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

def main(data_path: str):
	# Load dataset
	data = pd.read_csv(data_path)

	# Preview data (small sample)
	print("First 5 rows of dataset:")
	print(data.head())

	# Feature selection
	X = data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
	y = data['Clicked on Ad']

	# Train-test split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

	# Train model
	model = LogisticRegression()
	model.fit(X_train, y_train)

	# Predictions
	y_pred = model.predict(X_test)

	# Evaluate model
	print("\n--- Model Performance ---")
	print("Accuracy:", accuracy_score(y_test, y_pred))
	print("Precision:", precision_score(y_test, y_pred))
	print("Recall:", recall_score(y_test, y_pred))
	print("\nClassification Report:\n", classification_report(y_test, y_pred))

	# Confusion matrix
	cm = confusion_matrix(y_test, y_pred)
	sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
	plt.title("Confusion Matrix")
	plt.xlabel("Predicted")
	plt.ylabel("Actual")

	# Save figure to file so this runs in headless environments
	out_path = os.path.join(os.path.dirname(data_path), "confusion_matrix.png")
	plt.savefig(out_path)
	print(f"Confusion matrix saved to: {out_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train a logistic regression on advertising.csv and show metrics")
	parser.add_argument("--data", "-d", default="advertising.csv", help="Path to advertising CSV file")
	args = parser.parse_args()

	# Validate file exists
	if not os.path.exists(args.data):
		raise SystemExit(f"Data file not found: {args.data}")

	main(args.data)
