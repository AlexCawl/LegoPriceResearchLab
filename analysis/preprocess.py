import random
from typing import Tuple, Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from util.measurement import measure_execution_time


@measure_execution_time
def preprocess(lego: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    lego = lego.drop(columns=['Sets URL', 'Part URL'], axis=1, inplace=False)
    lego = lego.dropna(axis=0, inplace=False)
    lego['Star rating'] = lego['Star rating'].apply(lambda x: x.replace(',', '.'))

    numeric_features = ['Set Price', 'Number of reviews', 'Star rating', 'year']
    for feature in numeric_features:
        lego[feature] = pd.to_numeric(
            lego[feature].apply(lambda x: x.replace(",", "") if type(x) not in (int, float) else x))

    encoders_table: Dict[Any, LabelEncoder] = {}
    text_features = [feature for feature in lego.columns if feature not in numeric_features]

    for feature in text_features:
        encoders_table[feature] = LabelEncoder()
        lego[feature] = encoders_table[feature].fit_transform(lego[feature])
    lego.drop_duplicates(inplace=True, ignore_index=True)

    # prices = lego['Set Price']
    # prices = prices.apply(lambda x: x + random.randint(-5000, 5000))
    # lego['Set Price'] = prices

    return lego, encoders_table


@measure_execution_time
def split(lego: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(lego, test_size=0.33)
    x_train = train.drop(columns=["Set Price"], inplace=False)
    y_train = train[["Set Price", ]]
    x_test = test.drop(columns=["Set Price"], inplace=False)
    y_test = test[["Set Price", ]]
    return x_train, y_train, x_test, y_test
