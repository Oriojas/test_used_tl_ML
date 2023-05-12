from etl import Etl

FOLDER = "data/MLA_100k_checked_v3.jsonlines"
N = -2

X_train, y_train, X_test, y_test = Etl(folder_data=FOLDER, n=N).extract()

print(X_train)