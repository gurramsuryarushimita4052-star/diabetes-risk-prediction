
import pandas as pd

# Load the diabetes dataset
df = pd.read_csv("dataset/pima-indians-diabetes.csv")

print("DATASET LOADED SUCCESSFULLY!")

print("\nFIRST FIVE ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nDATA SUMMARY")
print(df.describe())

print("\nEDA COMPLETED SUCCESSFULLY!")