# Project Title

Machine Learning Trading Bot 

# Libraries and neccessary dependencies 

import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report
from sklearn import tree

# Background

To goal of this project is to demonstrate how ML can improve benchmark trading strategies as 
well as demontrate how some ML algo's are better than others. 

# Features of the Notebook

Establish a Baseline Performance

Tune the Baseline Trading Algorithm

Evaluate a New Machine Learning Classifier

Create an Evaluation Report

# Step 1

I established a baseline using a 4 SMA and a 100 SMA crossover strategy. The returns not very good. Starter_Code/bokeh_plot (5).png

I then adjusted the SMA's to 13 for the short window and and 30 for the long window. This improved the results slightly but not much. Starter_Code/bokeh_plot (7).png

# Step 2

I then evaluated ML using SVM model and compared its performance to the baseline performance. 
Starter_Code/bokeh_plot (9).png and it returned better results than the daily returns. 

I also adjusted the time frame from 3 months to 6 months. The ML algo performed better over a longer period of time. Starter_Code/machine_learning_trading_bot.ipynb

# Step 3

I then tried a new ML model called DecisionTree and this algo performed poorly compared to the
first SVM model. Starter_Code/machine_learning_trading_bot.ipynb


# Conclusion

With the provided dataset the SVM algo performs better over a longer period of time compared to other ML models like DecisionTrees and baseline strategies without ML. 
