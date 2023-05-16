import pandas as pd

FOLDER = "data/"

# df_split = pd.read_pickle(FOLDER + "df_text_clean.pkl")
#
# df_split = df_split.sample(1000)
#
# x = df_split["clean_text"]
# y = df_split["Y"]
#
# X_train, X_test, y_train, y_test = train_test_split(x,
#                                                     y,
#                                                     test_size=0.2,
#                                                     random_state=134)

FOLDER = "data/"

df_split = pd.read_pickle(FOLDER + "df_text_clean.pkl")

df_split = df_split.sample(1000)

df_train = df_split[0:-1]
df_test = df_split[81:]

df_train[["title", "label"]].to_csv(FOLDER + "df_text_train.csv", index=False)
df_test[["title", "label"]].to_csv(FOLDER + "df_text_test.csv", index=False)



print("ok")