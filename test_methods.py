import ast
import pandas as pd
from extract import Ext


def extrac_name(df_n, column, name_column):

    df_name = df_n[column].apply(lambda x: ast.literal_eval(str(x)))
    df_clean = []
    for i in range(len(df_n)):
        temp_1 = df_name.iloc[i][name_column]
        df_clean.append(temp_1)
    df_n[f"{column}_{name_column}"] = df_clean
    return df_n


FOLDER = "data/MLA_100k_checked_v3.jsonlines"
N = -2

X_train, y_train, X_test, y_test = Ext(folder_data=FOLDER, n=N).extract()

X_test = pd.DataFrame(X_test)

X_test = extrac_name(df_n=X_test, column="seller_address", name_column="country")
X_test = extrac_name(df_n=X_test, column="seller_address_country", name_column="name")
X_test = extrac_name(df_n=X_test, column="seller_address", name_column="state")
X_test = extrac_name(df_n=X_test, column="seller_address_state", name_column="name")
X_test = extrac_name(df_n=X_test, column="seller_address", name_column="city")
X_test = extrac_name(df_n=X_test, column="seller_address_city", name_column="name")

X_train = pd.DataFrame(X_train)

X_train = extrac_name(df_n=X_train, column="seller_address", name_column="country")
X_train = extrac_name(df_n=X_train, column="seller_address_country", name_column="name")
X_train = extrac_name(df_n=X_train, column="seller_address", name_column="state")
X_train = extrac_name(df_n=X_train, column="seller_address_state", name_column="name")
X_train = extrac_name(df_n=X_train, column="seller_address", name_column="city")
X_train = extrac_name(df_n=X_train, column="seller_address_city", name_column="name")

columns = X_train.columns

view_columns = ['seller_address', 'warranty', 'sub_status', 'condition', 'deal_ids',
                'base_price', 'shipping', 'non_mercado_pago_payment_methods',
                'seller_id', 'variations', 'site_id', 'listing_type_id', 'price',
                'attributes', 'buying_mode', 'tags', 'listing_source', 'parent_item_id',
                'coverage_areas', 'category_id', 'descriptions', 'last_updated',
                'international_delivery_mode', 'pictures', 'id', 'official_store_id',
                'differential_pricing', 'accepts_mercadopago', 'original_price',
                'currency_id', 'thumbnail', 'title', 'automatic_relist', 'date_created',
                'secure_thumbnail', 'stop_time', 'status', 'video_id',
                'catalog_product_id', 'subtitle', 'initial_quantity', 'start_time',
                'permalink', 'sold_quantity', 'available_quantity',
                'seller_address_country', 'seller_address_country_name',
                'seller_address_state', 'seller_address_state_name',
                'seller_address_city', 'seller_address_city_name']

X_train.to_csv("data/train.csv")
X_test.to_csv("data/test.csv")
