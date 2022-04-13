import numpy as np
import pandas as pd
import dask.dataframe as dd
import datetime
import functools

import sklearn.base as skbase
from category_encoders.leave_one_out import LeaveOneOutEncoder

from typing import List, Union, Optional, Dict
import featurelib as fl
# from featurelib import functional_transformer

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


class DayOfWeekReceiptsCalcer(fl.DateFeatureCalcer):
    name = 'day_of_week_receipts'
    keys = ['client_id']
    
    def __init__(self, delta: int, **kwargs):
        self.delta = delta
        super().__init__(**kwargs)

    def compute(self) -> dd.DataFrame:
        receipts = self.engine.get_table('receipts')

        date_to = datetime.datetime.combine(self.date_to, datetime.datetime.min.time())
        date_from = date_to - datetime.timedelta(days=self.delta)
        date_mask = (
                        (receipts['transaction_datetime'] >= date_from) & 
                        (receipts['transaction_datetime'] < date_to) & 
                        (receipts['purchase_sum'] > 0)
        )

        features = (
            receipts
            .loc[date_mask, ['client_id', 'transaction_datetime', 'purchase_sum']]
            .assign(dayofweek=lambda x: x['transaction_datetime'].dt.dayofweek)
            .categorize(columns=['dayofweek'])
        ).pivot_table(index=self.keys[0], 
                      columns='dayofweek', 
                      values='purchase_sum', 
                      aggfunc='count'
                     )
        
        orderedcols = features.columns.categories.values
        features = features[orderedcols]
        
        column_names = [
            f'purchases_count_dw{categ}__{self.delta}d' 
            for categ in orderedcols
        ]
        features.columns = column_names
        features = features.reset_index()
        
        return features
    
    
class FavouriteStoreCalcer(fl.DateFeatureCalcer):
    name = 'favourite_store'
    keys = ['client_id'] 
            
    def __init__(self, delta: int, **kwargs):
        self.delta = delta
        super().__init__(**kwargs)

#     @staticmethod        
#     def getFavourStore(x):
#         visits2stores = {}
#         stores = x['store_id'].value_counts().to_dict()
#         for k,v in stores.items():
#             visits2stores.setdefault(v, []).append(k)
#         max_visited = np.array(list(visits2stores.keys())).max()       
#         res = max(visits2stores[max_visited]) 
#         return pd.Series(res, index=["favourite_store_id"])           
            
    def compute(self) -> dd.DataFrame:
        receipts = self.engine.get_table('receipts')
        date_to = datetime.datetime.combine(self.date_to, datetime.datetime.min.time())
        date_from = date_to - datetime.timedelta(days=self.delta)
        date_mask = (
                        (receipts['transaction_datetime'] >= date_from) & 
                        (receipts['transaction_datetime'] < date_to) & 
                        (receipts['purchase_sum'] > 0)
        )         

        features = receipts.loc[date_mask, self.keys+['store_id']]  
        
        features = dask_groupby(
            features,
            by=['client_id','store_id'],
            config={
                "store_id": ["count"]
            }
        )             
        features = features.reset_index()
        
        maxmask = dask_groupby(
            features,
            by=['client_id'],
            config={
                "store_id_count": ["max"]
            }
        )         
        maxmask = maxmask.reset_index()
        maxmask = maxmask.drop_duplicates(['client_id', 'store_id_count_max'])

        features = \
        features.merge(
                maxmask[['client_id', 'store_id_count_max']],
                on=['client_id'],
                how='inner'
            )
        
        date_mask = (features['store_id_count'] == features['store_id_count_max'])
        features = features.loc[date_mask,['client_id','store_id']]

        features = dask_groupby(
            features,
            by=['client_id'],
            config={
                "store_id": ["max"]
            }
        )             
        features = features.reset_index()        
        features = features.rename(columns={"store_id_max": f"favourite_store_id__{self.delta}d"})
        return features
        
        

# class TargetFromCampaignsCalcer(fl.DateFeatureCalcer):
#     name = 'target_from_campaigns'
#     keys = ['client_id']
    
#     def compute(self) -> dd.DataFrame:
#         campaigns = self.engine.get_table('campaigns')
#         date_mask = (dd.to_datetime(campaigns['treatment_date'], format='%Y-%m-%d').dt.date == self.date_to)

#         result = (
#             self.engine.get_table('campaigns')
#             .loc[date_mask]
#             [[
#                 'client_id', 'treatment_flg',
#                 'target_purchases_sum', 'target_purchases_count', 'target_campaign_points_spent'
#             ]]
#         )
#         return result    
    
# class AgeGenderCalcer(fl.FeatureCalcer):
#     name = 'age_gender'
#     keys = ['client_id']

#     def compute(self) -> dd.DataFrame:
#         client_profile = self.engine.get_table('client_profile')
#         return client_profile[self.keys + ['age', 'gender']]
    
# class ExpressionTransformer(skbase.BaseEstimator, skbase.TransformerMixin):
#     expression: str 
#     col_result: str

#     def __init__(self, function, **params):
#         self.function = functools.partial(function, **params)

#     def fit(self, *args, **kwargs):
#         return self

#     def transform(self, *args, **kwargs) -> pd.DataFrame:
#         return self.function(*args, **kwargs) 

class LOOMeanTargetEncoder(skbase.BaseEstimator, skbase.TransformerMixin):
         
    def __init__(self, **params):
        self.col_categorical, self.col_target, self.col_result = params.values()
#         self.function = functools.partial(function, **params)
        self.encoder_ = LeaveOneOutEncoder(cols=[self.col_categorical])
    
    def fit(self, data, *args, **kwargs):
            y = None
            if self.col_target in data.columns:
                y = data[self.col_target]
            self.encoder_.fit(data[self.col_categorical], y=y)
            return self

    def transform(self, data, *args, **kwargs) -> pd.DataFrame:
        y = None
        if self.col_target in data.columns:
            y = data[self.col_target]
        data[self.col_result] = self.encoder_.transform(data[self.col_categorical], y=y)
        return data    
    
# def functional_transformer(function):
#     def builder(**params):
#         return ExpressionTransformer(function, **params)
#     return builder 

# def looe_transformer(function):
#     def builder(**params):
#         return LOOMeanTargetEncoder(function, **params)
#     return builder 

@fl.functional_transformer
def ExpressionTransformer(data: pd.DataFrame, expression: str, col_result: str):
    col_result = col_result
    df_name = "data"
    data[col_result] = eval(expression.format(d=df_name))
    return data

# @looe_transformer
# def transform_cols_looe(data: pd.DataFrame, col_categorical: str, col_target: str, col_result: str):
#     return data


# TABLES = {
#     'receipts': receipts,
#     'campaigns': campaigns,
#     'client_profile': client_profile,
#     'products': products,
#     'purchases': purchases,
# }

# engine = fl.Engine(tables=TABLES)

# fl.register_calcer(DayOfWeekReceiptsCalcer)
# fl.register_calcer(FavouriteStoreCalcer)
# fl.register_calcer(TargetFromCampaignsCalcer)
# fl.register_calcer(AgeGenderCalcer)

# fl.register_transformer(transform_cols, 'expression')
# fl.register_transformer(transform_cols_looe, 'loo_mean_target_encoder')