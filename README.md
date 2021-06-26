# Multistep Input-Output Timeseries using LSTM
Python script for regression of multivariable LSTM neural network.

## Requirements
- python == 3.9.1
- tensorflow == 2.5.0
- pandas == 1.2.4
- numpy == 1.19.5
- seaborn == 0.11.1
- matplotlib == 3.4.2
- scikit-learn == 0.24.2
- keras == 2.4.3

## Input Specifications
- Sample dataset is contained as `data.csv`
- `inputs` number of inputs for model training/testing
- Number of hidden `neurons` per LSTM and FC layers
- Moving Window of `m` number of time step inputs
- `n` day output Multistep forecasting

## Script Specifications
- Load time series dataset CSV with specified (variables `inputs` inputs) – denoted in the sample dataset.
- Input preprocessed (StandardScalar) and using TimeSeriesSplit Cross-Validation
- Each LSTM model architecture has:
  - 2x LSTM layer (with their “number of hidden `neurons`” as variables) followed by 1x FC 
  - Specifications:
 
    i. Moving Window Time Step Input i.e. where x multivariable inputs {x(t-m)…x(t-1) and x(t)}, where `m` is a variable
    
    ii. EarlyStopping Callback with patience = 20 during the training phase
    
    iii. Loss function: MSE
- Prediction phase to predict `n` days where y is output: y(t),y(t+1),…y(t+n)
-  A multi-step approach, and where `n` is a variable

## Implementation
Make changes in this part of the script to customise it to your dataset
```
obj = TSLSTM(
        path_to_csv="./data.csv", 
        inputs=[0,1,2,3,4,5], 
        neurons=50, 
        m=2, 
        n=1,
        approach="lstmd"
        )
```
To run the script, simply use `python tslstm.py`

## References
- [Multistep Time Series Forecasting with LSTMs in Python](https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/)
- [Multi-Step Multivariate Time-Series Forecasting using LSTM](https://pangkh98.medium.com/multi-step-multivariate-time-series-forecasting-using-lstm-92c6d22cd9c2)

## Contributions
I am looking forward to implementing the following 2 papers:
1. [BAYESIAN RECURRENT NEURAL NETWORKS](https://arxiv.org/pdf/1704.02798.pdf)
2. [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/pdf/1506.02142.pdf)

Any contributions to the current script or in implementing the above 2 papers are welcome.
