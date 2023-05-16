import pandas as pd
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

df = pd.read_pickle("data/df_num_norm.pkl")
# df = df.sample(n=80000)

TargetVariable = 'bin_class'
Predictors = ['listing_type_id_bronze', 'listing_type_id_free',
              'listing_type_id_gold', 'listing_type_id_gold_premium',
              'listing_type_id_gold_pro', 'listing_type_id_gold_special',
              'listing_type_id_silver', 'available_quantity_False', 'price_norm']

X = df[Predictors].values
y = df[TargetVariable].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=137)

clf = XGBClassifier(max_depth=3,
                    learning_rate=0.01,
                    n_estimators=500,
                    objective='binary:logistic',
                    booster='gbtree')


print(clf)


XGB = clf.fit(X_train, y_train)
prediction = XGB.predict(X_test)

print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

feature_importances = pd.Series(XGB.feature_importances_,
                                index=Predictors)
feature_importances.nlargest(10).plot(kind='barh')

# Printing some sample values of prediction
TestingDataResults = pd.DataFrame(data=X_test, columns=Predictors)
TestingDataResults['TargetColumn'] = y_test
TestingDataResults['Prediction'] = prediction
TestingDataResults.head()
