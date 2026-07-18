import pandas as pd 

DATA_PATH = "data/creditcard.csv"

df = pd.read_csv(DATA_PATH)

print("Satır Sayısı:",df.shape[0])
print("Sütun Sayısı:",df.shape[1])

print("\nİlk 5 satır:")
print(df.head())

print("\nClass dağılımı:")
print(df["Class"].value_counts())

print("\nClass yüzdesi:")
print(df["Class"].value_counts(normalize=True) * 100)


print("\nEksik değer sayısı:")
print(df.isnull().sum().sum())

