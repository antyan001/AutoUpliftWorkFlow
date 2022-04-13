# Automated and Configurable Feature Calcer Engine in connection with UPLIFT Modelling 

*PROJECT STRUCTURE*:
- `AutoUpliftWorkFlow`:  
    - `MainCalcer.ipynb`: main jupyter notebook incorporating: 
    1. all preproc-transformers and automated Feature Calcer pipes
    1. various Uplift modelling strategies
    - `datalib/features/`:
        - `base.py`: derived main abstract classes --> `DateFeatureCalcer`, `FeatureCalcer`
        - `compute.py`: `create_pipeline` --> parse instructions from user `config` (defined in `MainCalcer.ipynb`) and join tables following with register_transformer (`ExpressionTransformer`, `BinningTransformer`, `OneHotEncoder`)
        - `extract.py`: implemented various aggregation strategies based in main `dask_groupby` class
        - `transform.py`: `DivideColsTransformer`, `BinningTransformer`, `ExpressionTransformer`
    - `campaign_flow.py`: derived main class `create_transform_pipeline` to run all steps in one pipe
    
- `BuildFeatureCalcer`:
    - `BuildFeatureCalcer.ipynb`: main jupyter notebook incorporating all blocks for whole FeatureCalcer Pipeline.
    - `/src`:
        - `featurelib.py`: library contains --> `create_calcer`, `compute_features`, `build_pipeline` functions
        - `feature_impl.py`: contains child classes derived from abstract `DateFeatureCalcer` and `FeatureCalcer` classes:
            1. `AgeGenderCalcer` 
            1. `ReceiptsBasicFeatureCalcer`
            1. `TargetFromCampaignsCalcer`
            1. `OneHotEncoder`      
    - `data_config`: provide all aggregation/transformation instructions for FeatureCalcer Pipe
    - `FeatureCalcer.py`: derived feature agg classes `DayOfWeekReceiptsCalcer`, `FavouriteStoreCalcer`, `ExpressionTransformer`  
