import math

import pandas as pd
from fastai.data.transforms import FuncSplitter, Normalize
from fastai.tabular.core import TabularPandas, FillMissing, Categorify, CategoryBlock, TabDataLoader, DataLoaders, ifnone
from fastai.tabular.learner import tabular_learner, F1Score, accuracy

from ams.config import logger_factory
from ams.utils import date_utils

logger = logger_factory.create(__name__)

def test_df():
    procs = [FillMissing, Categorify, Normalize]
    cat_vars = ['c']
    cont_vars = ['b']
    df = pd.DataFrame({'a': ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-10"], 'b': [0, 0, 0, 0, 1], 'c': ['blue', 'black', 'orange', 'red', 'yellow']})

    a_dist = df['a'].unique().tolist()
    num_dates = len(a_dist)
    pct_train = 80

    train_cut_off_ndx = math.floor(num_dates * (pct_train / 100)) - 1
    train_cut_off = a_dist[train_cut_off_ndx]
    logger.info(f"tcoi: {train_cut_off_ndx}; {train_cut_off}")

    def split(dt_str):
        return True if train_cut_off > dt_str else False

    splits = FuncSplitter(split)(df['a'])

    to = TabularPandas(df, procs, cat_vars, cont_vars, y_names="c", y_block=CategoryBlock(),
                       splits=splits, do_setup=True)
    trn_dl = TabDataLoader(to.train, bs=64, num_workers=0, shuffle=True, drop_last=True)
    val_dl = TabDataLoader(to.valid, bs=128, num_workers=0)

    dls = DataLoaders(trn_dl, val_dl).cuda()

    # learn = tabular_learner(dls, metrics=F1Score(average="macro"))
    learn = tabular_learner(dls, metrics=accuracy)

    lr = learn.lr_find()

    learn.fit_one_cycle(10, lr_max=lr)

    dt = date_utils.parse_std_datestring(date_str)
    dt.weekday() in [5, 6]

    # def emb_sz_rule(n_cat):
    #     "Rule of thumb to pick embedding size corresponding to `n_cat`"
    #     return min(600, round(1.6 * n_cat ** 0.56))
    #
    # def _one_emb_sz(classes, n, sz_dict=None):
    #     "Pick an embedding size for `n` depending on `classes` if not given in `sz_dict`."
    #     sz_dict = ifnone(sz_dict, {})
    #     n_cat = len(classes[n])
    #     sz = sz_dict.get(n, int(emb_sz_rule(n_cat)))  # rule of thumb
    #     return n_cat, sz
    #
    # def get_emb_sz(to, sz_dict=None):
    #     "Get default embedding size from `TabularPreprocessor` `proc` or the ones in `sz_dict`"
    #     return [_one_emb_sz(to.procs.classes, n, sz_dict) for n in to.cat_names]
    #
    # emb_szs = get_emb_sz(to)


