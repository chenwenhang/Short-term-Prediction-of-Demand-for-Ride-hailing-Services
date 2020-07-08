# Short-term prediction of demand for ride-hailing services

### Introduction
The project uses Time Series, Lasso Regression, LightGBM, Ridge Regression and DNN respectively to predict the demand for ride-hailing services.

### Result
##### Performance on yellow taxi dataset
| **Model Name** | **R2 Score** |
| --- | --- |
| DNN | 0.9386 |
| Ridge Regression | 0.8568 |
| Lasso Regression | 0.8417 |
| LightGBM | 0.0019 |

##### Performance on bike share dataset
| **Model Name** | **R2 Score** |
| --- | --- |
| DNN | 0.6115 |
| Time Series | 0.5689 |
| Ridge Regression | 0.3963 |
| Lasso Regression | 0.2478 |
| LightGBM | 0.1260 |

### Dataset
[Yellow-taxi data source](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)  
[Yellow-taxi data description](https://github.com/chenwenhang/Short-term-Prediction-of-Demand-for-Ride-hailing-Services/tree/master/data)  
[Bike share data source & description](https://www.kaggle.com/akkithetechie/new-york-city-bike-share-dataset)