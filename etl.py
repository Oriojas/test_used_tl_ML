import json


class Etl:

    def __init__(self, folder_data: str, n: int):
        self.folder_data = folder_data
        self.N = n

    def extract(self):
        N = self.N
        data = [json.loads(x) for x in open(self.folder_data)]
        target = lambda x: x.get("condition")

        X_train = data[:N]
        X_test = data[N:]

        y_train = [target(x) for x in X_train]
        y_test = [target(x) for x in X_test]

        for x in X_test:
            del x["condition"]

        return X_train, y_train, X_test, y_test
