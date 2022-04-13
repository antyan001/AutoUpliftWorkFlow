# User Defined Feature Calculation on Hadoop and Training Boosting Models using PySpark Broadcasting  

*PROJECT STRUCTURE*:
- `GoogleAnalyticsClickstreamPreproc`: GA notebooks revealing UDF for `features calculation` based on user' clickstream data: FE with monthly, weekly, daily frequencies  
- `AutoEmailResponceModel`: project for choosing best emailing strategy based on user responses (`XGboost`/`LightGBM` with pySpark jobs)
    - `CronPy`: all in one py-scripts for scheduling jobs
    - `FinalPredict`: `features calculation`, batch retraining strategies and boosting inference with `pySpark` 
    - `ModelFineTuneAndFeaturesSelection`: Fine-Tuned strategies for Model Training   
    - `PCA_TSNE_GaussMix`: performing `PCA`/`TSNE` with GaussianMixture Modelling
- `/src/`: different scripts for preprocessing input data
- `support_library`: usefull pyclass-scripts for FE, `FeaturePermutations`, FW/BW `Feature Elimination`, `Label Encodings`, `WOE`  
