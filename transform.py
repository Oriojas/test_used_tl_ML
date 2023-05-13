import json
import pandas as pd



class Trans:

    def __init__(self, X_train, y_train, X_test, y_test):
        X_train, y_train, X_test, y_test = self.X_train, self.y_train, self.X_test, self.y_test