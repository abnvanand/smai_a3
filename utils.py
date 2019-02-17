import random

def train_test_split(X, y, test_size, random_state=None):
    """Splits data into training and testing sets.
    test_size must be fractional eg 0.2 for 20% split"""

    if random_state is not None:
        # Seed to generate same pseudo-random numbers everytime to make it reproducible.
        random.seed(random_state)

    test_size = round(test_size * len(X))  # change proportion to actual number of rows

    indices = X.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    X_test = X.loc[test_indices, :]
    y_test = y.loc[test_indices] 
    X_train = X.drop(test_indices)
    y_train = y.drop(test_indices)

    return X_train, X_test, y_train, y_test

