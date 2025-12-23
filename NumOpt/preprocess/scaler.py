import numpy as np
from ..cprint import cprint_red
from ..opti import cas


class NormalScaler:
    __slots__ = ("_mean", "_std", "_target_mean", "_target_std")

    def __init__(self, mean=None, std=None):
        self._mean = mean
        self._std = std
        self._target_mean = None
        self._target_std = None

    @staticmethod
    def fit(dataset):
        if dataset.ndim != 2:
            cprint_red("The dataset ndim must = 2.\n")
            exit(-1)
        mean = np.mean(dataset, axis=0, keepdims=True)
        std = np.std(dataset, axis=0, ddof=1, keepdims=True)
        return NormalScaler(mean, std)

    def set_target(self, target_mean, target_std):
        self._target_mean = np.array(target_mean).reshape((1, -1))
        self._target_std = np.array(target_std).reshape((1, -1))

    def transform(self, dataset):
        nrows = dataset.shape[0]
        old_mean = cas.repmat(self._mean, nrows, 1)
        old_std = cas.repmat(self._std, nrows, 1)
        new_mean = cas.repmat(self._target_mean, nrows, 1)
        new_std = cas.repmat(self._target_std, nrows, 1)

        new_dataset = (dataset - old_mean) / old_std
        new_dataset = new_dataset * new_std + new_mean
        return new_dataset

    def inverse_transform(self, dataset):
        nrows = dataset.shape[0]
        old_mean = cas.repmat(self._mean, nrows, 1)
        old_std = cas.repmat(self._std, nrows, 1)
        new_mean = cas.repmat(self._target_mean, nrows, 1)
        new_std = cas.repmat(self._target_std, nrows, 1)

        new_dataset = (dataset - new_mean) / new_std
        new_dataset = new_dataset * old_std + old_mean
        return new_dataset

    def __getstate__(self):
        states = {}
        for key in self.__slots__:
            states[key] = getattr(self, key)
        return states

    def __setstate__(self, states):
        for key, val in states.items():
            setattr(self, key, val)


if __name__ == "__main__":
    import pickle
    import os

    x = np.random.normal(0.0, 0.1, (1000, 1))

    scaler = NormalScaler().fit(x)
    scaler.set_target([0.0, 0.0, 0.0], [100.0, 100.0, 100.0])

    new_x = scaler.transform(dataset=x)

    print(scaler.__getstate__())
    print(np.mean(new_x, axis=0))

    with open("test.pkl", "wb") as fout:
        pickle.dump(scaler, fout)

    with open("test.pkl", "rb") as fin:
        obj = pickle.load(fin)
        print(obj.__getstate__())

    os.unlink("test.pkl")
