from sklearn.model_selection import train_test_split


def split_train_validation(X, y, val_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=val_size, random_state=random_state)
