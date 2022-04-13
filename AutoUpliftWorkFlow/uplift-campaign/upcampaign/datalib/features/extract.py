
import dask.dataframe as dd
import pandas as pd
import datetime
import sklearn.base as skbase
import sklearn.preprocessing as skpreprocessing

from typing import List, Dict, Union

from .base import DateFeatureCalcer, FeatureCalcer


def dask_groupby(
    data: dd.DataFrame,
    by: List[str],
    config: Dict[str, Union[str, List[str]]]
) -> dd.DataFrame:
    data_ = data.copy()
    dask_agg_config = dict()

    for col, aggs in config.items():
        aggs = aggs if isinstance(aggs, list) else [aggs]
        for agg in aggs:
            fictious_col = f'{col}_{agg}'
            data_ = data_.assign(**{fictious_col: lambda d: d[col]})
            dask_agg_config[fictious_col] = agg

    result = data_.groupby(by=by).agg(dask_agg_config)
    return result


class AgeGenderCalcer(FeatureCalcer):
    name = 'age_gender'
    keys = ['customer_id']

    def compute(self) -> dd.DataFrame:
        client_profile = self.engine.get_table('client_profile')
        return client_profile[self.keys + ['age', 'location']]


class ReceiptsBasicFeatureCalcer(DateFeatureCalcer):
    name = 'receipts_basic'
    keys = ['customer_id']

    def __init__(self, delta: int, **kwargs):
        self.delta = delta
        super().__init__(**kwargs)

    def compute(self) -> dd.DataFrame:
        receipts = self.engine.get_table('receipts')
        date_to = self.date_to
        date_from = self.date_to - self.delta
        date_mask = (receipts['date'] >= date_from) & (receipts['date'] < date_to)

        features = (
            receipts
            .loc[date_mask]
            .assign(purchase_sum_w_disc=lambda d: d['purchase_sum'] - d['discount'])
            .assign(discount_flag=lambda d: (d['discount'] > 0).astype(int))
        )
        features = dask_groupby(
            features,
            by=['customer_id'],
            config={
                "purchase_amt": ["sum", "max", "min", "mean"],
                "purchase_sum": ["sum", "max", "min", "mean"],
                "purchase_sum_w_disc": ["sum", "max", "min", "mean"],
                "date": ["min", "max", "count"],
                "discount_flag": ["sum"]
            }
        )
        features = (
            features
            .assign(
                mean_time_interval=lambda d: (
                    (d['date_max'] - d['date_min'])
                    / (d['date_count'])
                )
            )
            .assign(
                time_since_last=lambda d: (
                    date_to - d['date_max']
                )
            )
        )

        features = features.reset_index()
        features = features.rename(columns={
            col: col + f'__{self.delta}d' for col in features.columns if col not in self.keys
        })

        return features


class TargetFromCampaignsCalcer(DateFeatureCalcer):
    name = 'target_from_campaigns'
    keys = ['customer_id']
    
    def compute(self) -> dd.DataFrame:
        campaigns = self.engine.get_table('campaigns')
        date_mask = (campaigns.date == self.date_to)

        result = (
            self.engine.get_table('campaigns')
            .loc[date_mask]
            [[
                'customer_id', 'target_group_flag'
            ]]
        )
        return result
