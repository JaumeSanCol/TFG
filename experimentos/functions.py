from sklearn.datasets import load_iris, load_digits
from keras.datasets import mnist, fashion_mnist
import pandas as pd

def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min >= -128 and c_max <= 127:
                    df[col] = df[col].astype('int8')
                elif c_min >= -32768 and c_max <= 32767:
                    df[col] = df[col].astype('int16')
                elif c_min >= -2147483648 and c_max <= 2147483647:
                    df[col] = df[col].astype('int32')
                else:
                    df[col] = df[col].astype('int64')
            else:
                df[col] = df[col].astype('float32') 
    return df

def load_dataset(name):
    if name == 'Iris':
        data = load_iris()
        X, y = data.data, data.target

    elif name == 'Digits':
        data = load_digits()
        X, y = data.data, data.target

    elif name == 'MNIST':
        (X_train, y_train), _ = mnist.load_data()
        X = X_train.reshape((X_train.shape[0], -1))
        y = y_train

    elif name == 'Fashion MNIST':
        (X_train, y_train), _ = fashion_mnist.load_data()
        X = X_train.reshape((X_train.shape[0], -1))
        y = y_train

    else:
        raise ValueError(f"Dataset {name} no reconocido.")

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X = reduce_mem_usage(X)

    return X.values, y