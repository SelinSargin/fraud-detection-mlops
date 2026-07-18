import pandas as pd
from  sklearn.model_selection import train_test_split

DATA_PATH = "data/creditcard.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y


def split_data(X,y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, 
        y, 
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def print_class_ratio(name,y_data):
    print(f"{name} ")
    print(y_data.value_counts())
    print(y_data.value_counts(normalize=True)* 100)
    

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X,y)

    print("X_Train",X_train.shape)
    print("x_val",X_val.shape)
    print("x_test",X_test.shape)

    print_class_ratio("Train class dağılımı",y_train)
    print_class_ratio("Val class dağılımı",y_val)
    print_class_ratio("Test class dağılımı",y_test)

