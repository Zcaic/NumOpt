import numpy as np
from ..cprint import cprint_green, cprint_red

class NormalScaler:
    def __init__(self):
        self._mean = None
        self._std = None

    def fit(self, dataset):
        if dataset.ndim != 2:
            cprint_red("The dataset ndim must = 2.\n")
            exit(-1)
        self._mean = np.mean(dataset, axis=0)
        self._std = np.std(dataset, axis=0, ddof=1)
        return self

    def transform(self, dataset, target_mean, target_std):
        new_dataset = (dataset - self._mean) / self._std
        new_dataset = new_dataset * target_std + target_mean
        return new_dataset


if __name__ == "__main__":
    x = np.random.normal(0.0, 0.1, (1000, 1))

    scaler = NormalScaler().fit(x)
    print(scaler._mean, scaler._std)

    new_x = scaler.transform(dataset=x, target_mean=2.0, target_std=10)
    new_scaler=NormalScaler().fit(new_x)
    print(new_scaler._mean,new_scaler._std)