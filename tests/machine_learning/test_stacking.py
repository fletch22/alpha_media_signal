import warnings
from statistics import mean

import pandas as pd
import xgboost as xgb
from matplotlib import pyplot
from numpy import std
from sklearn.datasets import make_classification
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# get the dataset
from ams.config import constants, logger_factory
from ams.services.twitter_service import convert_to_arrays, get_split_prepped_twitter_data
from ams.twitter.TwitterStackingModel import TwitterStackingModel

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

logger = logger_factory.create(__name__)


def get_dataset():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
    return X, y


def get_stacking(balance_ratio: float):
    level0 = list()
    level0.append(('xgb', get_xgb(balance_ratio=balance_ratio)))
    level0.append(('lr', LogisticRegression(max_iter=500, n_jobs=-1)))
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('cart', DecisionTreeClassifier()))
    level0.append(('svm', SVC()))

    # define meta learner model
    level1 = LogisticRegression()

    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model


def get_twitter_stacking(twit_stack_model: TwitterStackingModel):
    level0 = list()

    for ndx, model in enumerate(twit_stack_model.models):
        level0.append((f"xgb_{ndx}", model))

    # define meta learner model
    level1 = LogisticRegression()

    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

    return model


def get_xgb(balance_ratio: float):
    return xgb.XGBClassifier(seed=42, max_depth=8, use_label_encoder=False, scale_pos_weight=balance_ratio)


# get a list of models to evaluate
def get_models(balance_ratio: float):
    models = dict()
    with warnings.catch_warnings():
        msg = "Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior."
        warnings.filterwarnings("ignore", message=msg)
        models['xgb'] = get_xgb(balance_ratio=balance_ratio)

    models['lr'] = LogisticRegression(max_iter=500, n_jobs=-1)
    models['knn'] = KNeighborsClassifier()
    models['cart'] = DecisionTreeClassifier()
    models['svm'] = SVC()
    models['stacking'] = get_stacking(balance_ratio=balance_ratio)
    return models


# get a list of models to evaluate
def get_twitter_models(twit_stack_model: TwitterStackingModel):
    models = dict()

    for ndx, model in enumerate(twit_stack_model.models):
        models[f"xgb_{ndx}"] = model

    models['stacking'] = get_twitter_stacking(twit_stack_model=twit_stack_model)

    return models


# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


def decimate(df: pd.DataFrame):
    return df.sample(frac=.5)


def test_stacking():
    requires_balance = False
    df_train = pd.read_parquet(constants.SAMPLE_TWEET_STOCK_TRAIN_DF_PATH)
    df_test = pd.read_parquet(constants.SAMPLE_TWEET_STOCK_TEST_DF_PATH)

    df_train = decimate(df=df_train)

    df_train.loc[:, "buy_sell"] = df_train["buy_sell"].apply(lambda bs: 1 if bs == 1 else 0)

    df_train = df_train.fillna(value=0)

    balance_ratio = 1.
    if not requires_balance:
        num_buy = df_train[df_train["buy_sell"] == 1].shape[0]
        num_sell = df_train[df_train["buy_sell"] != 1].shape[0]

        balance_ratio = num_sell / num_buy

        logger.info(f"Sell: {num_sell} / Buy: {num_buy}; ratio: {balance_ratio}")

    print(f"Num rows: {df_train.shape[0]}")

    X_train, y_train, _ = convert_to_arrays(df=df_train, label_col="buy_sell", require_balance=False)
    # X_train, y_train = get_dataset()

    # get the models to evaluate
    models = get_models(balance_ratio=balance_ratio)

    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, X_train, y_train)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

    # plot model performance for comparison
    pyplot.boxplot(results, labels=names, showmeans=True)
    pyplot.show()


def test_tweet_stacking():
    requires_balance = False
    X_train, y_train, _, _ = get_split_prepped_twitter_data(requires_balance)

    twit_stock_model = TwitterStackingModel.load()

    # get the models to evaluate
    models = get_twitter_models(twit_stack_model=twit_stock_model)

    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, X_train, y_train)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

    # plot model performance for comparison
    pyplot.boxplot(results, labels=names, showmeans=True)
    pyplot.show()