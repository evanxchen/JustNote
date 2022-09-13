# 問題定義
> 在金融業的情境中，想要預測顧客對於未來資金的動用量，以便做資金資源的調控，或者未來針對有巨幅變動之客戶議價或者給予個人化的服務。
>>  Ｘ:[年齡、往來程度、違約情形、現階段交易量等等．．．]  <br/>
>>  Ｙ：資金動用量（或者增加與否）

但客戶動用量是動態的，會隨著該位客戶現階段的狀態而一直改變，而因為動用量會受到前一段時間金流使用量或者淡旺季節影響，因此具有時間序列的關係，因此想要使用時間序列模型來做。 <br/>

Ex: 比如一位顧客在n, n+1, n+2.....個月內有各[X1, X2, X3, ... ,Xn]等變量，會對應到不同的[Yn, Yn+1,....]動用量，全部又有Ｍ個顧客。

* Q1: 目前谷歌到的研究多半是針對單一樣本（即可能只有一個顧客）的長期量預測（且是Ｙ去預測Y的t+1)，不知道資料架構應該要怎麼寫。
* Q2: 有谷歌到一些作法，目前最有可能採用的是CNN-LSTM，應用於多變量、多個樣本的，變量特徵的方式好像可以透過AUTO-ENCODER的方式去降維，但是如果多個顧客樣本應該要怎麼訓練？
* > 谷歌的結果有人說可以用batch的方式去訓練，但想知道是否有別種方式？

* Q3:有找到一些很類似的[paper](https://www.mdpi.com/2073-4441/14/15/2377)在做一樣的應用，但是後來深入發現CNN好像是多半應用於圖層，不知道這種概念是不是一樣的？


---





# **自己整理的時間序列相關分析的筆記**  
> ### Tutorials ：
* [李宏毅－about AutoDecoders](https://hackmd.io/@overkill8927/SyyCBk3Mr?type=view#25-Unsupervised-Learning---Auto-EncoderDecoder)
* [LSTM 自编码器的温和介绍](https://github.com/apachecn/ml-mastery-zh/blob/master/docs/lstm/lstm-autoencoders.md)
* [PyTorch搭建LSTM](https://blog.csdn.net/Cyril_KI/article/details/123963061?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-4-123963061-blog-99886972.pc_relevant_aa_2&spm=1001.2101.3001.4242.3&utm_relevant_index=7)
* [Time Series 101](https://www.kaggle.com/code/thebrownviking20/everything-you-can-do-with-a-time-series)
* [LSTM 101](https://blog.csdn.net/weixin_39653948/article/details/105366425)
* LSTM 結構 
	>[LSTM timedistribute](https://blog.csdn.net/LaoChengZier/article/details/88706642) <br />
	>[LSTM 数据格式](https://blog.csdn.net/he_wen_jie/article/details/79982211)<br />
	>[machinery tutorials](https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/) <br/>
	>[CNN-LSTM](https://hackmd.io/@subject/BJWLeCSNd) <br/>
	>[CNN-LSTM2](https://blog.csdn.net/Cyril_KI/article/details/126578034)
* LSTM 一些概念
	>[LSTM batch learning2](https://stackoverflow.com/questions/65144346/feeding-multiple-inputs-to-lstm-for-time-series-forecasting-using-pytorch) <br/>
	>[batch learning2](https://www.reddit.com/r/MLQuestions/comments/rn4j5p/how_to_train_one_lstm_model_with_independent/) <br/>
	>[Stateful and stateless 1](https://fairyonice.github.io/Stateful-LSTM-model-training-in-Keras.html) <br>
	>[stateful and stateless 2](https://zhuanlan.zhihu.com/p/34495801)

* LSTM CASE
    >[Beijing PM2.5 Case](https://blog.csdn.net/weixin_42608414/article/details/99886972) <br/>
---

* ## 時序上的因果關係檢定- 證明是時間相依的變數

  - The Granger Causality test is used to determine whether or not one time series is useful for forecasting another.

```python
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 100)
```

```python
fb_sent = pd.read_csv('./data/fb_sent/fb_sent.csv')
snownlp_sent = pd.read_csv('./data/snownlp_sent/snownlp_sent.csv')
deepnltk = pd.read_csv('./data/deeplnltk/deeplnltk_proc.csv')
baidu_sent = pd.read_csv('./data/baidu_sent/baidu_sent_proc2.csv')
```
```python
sentiment = pd.merge(left=fb_sent, right=snownlp_sent, how='left', on='DATE')
sentiment = pd.merge(left=sentiment, right=deepnltk, how='left', on='DATE')
sentiment = pd.merge(left=sentiment, right=baidu_sent, how='left', on='DATE')

sentiment
```
```python
df = df.loc[df['DATE'].apply(lambda x: '2021-10-01' <= x <= '2022-03-31')]
df = df.set_index('DATE')
std = StandardScaler()
df = pd.DataFrame(std.fit_transform(df), columns=df.columns, index=df.index)
df
```
```python
#perform Granger-Causality test
print('======================= LIKE =======================')
result = grangercausalitytests(df[['CLOSING_INDEX', 'LIKE']], maxlag=5)
print('======================= HAHA =======================')
result = grangercausalitytests(df[['CLOSING_INDEX', 'HAHA']], maxlag=5)
print('======================= LOVE =======================')
result = grangercausalitytests(df[['CLOSING_INDEX', 'LOVE']], maxlag=5)
print('======================= WOW========================')
result = grangercausalitytests(df[['CLOSING_INDEX', 'WOW']], maxlag=5)
print('======================= CARE =======================')
result = grangercausalitytests(df[['CLOSING_INDEX', 'CARE']], maxlag=5)
print('======================= ANGRY =======================')
result = grangercausalitytests(df[['CLOSING_INDEX', 'ANGRY']], maxlag=5)
print('======================= SAD =======================')
result = grangercausalitytests(df[['CLOSING_INDEX', 'SAD']], maxlag=5)

print('======================= SnowNLP_sent =======================')
result = grangercausalitytests(df[['CLOSING_INDEX', 'SnowNLP_sent']], maxlag=5)

print('======================= DEEPNLTK_NEG =======================')
result = grangercausalitytests(df[['CLOSING_INDEX', 'DEEPNLTK_NEG']], maxlag=5)
print('======================= DEEPNLTK_POS =======================')
result = grangercausalitytests(df[['CLOSING_INDEX', 'DEEPNLTK_POS']], maxlag=5)

print('======================= BAIDU_NEG =======================')
result = grangercausalitytests(df[['CLOSING_INDEX', 'BAIDU_NEG']], maxlag=5)
print('======================= BAIDU_POS =======================')
result = grangercausalitytests(df[['CLOSING_INDEX', 'BAIDU_POS']], maxlag=5)

```
---
* ## **Augmented Dickey-Fuller Test for stationary**
   >[檢定自相關程度高低](https://www.kaggle.com/code/galibce003/stationarity-and-dickey-fuller-test-with-example/notebook)

***

## **Models for Times Series**

| samples data | variables | method| Aim |examples
| :--:| :--: | :--:| :-------:|  :-----:|
| multiple inputs / parellel inputs | univariate | autoencoder + MLP regression |predict future value per sample individually |[store sales predictions](https://www.kaggle.com/code/dimitreoliveira/time-series-forecasting-with-lstm-autoencoders/notebook)   <br />[Web Traffic Time Series Forecasting](https://www.kaggle.com/code/ganeshhalpatrao/web-traffic-time-series-forecasting)
| One input| multiivariate | CNN.autoencoder + keras.LSTM| predict future value with given multiple variables into concerns| [LSTM Models for multi-step time series forcast](https://www.kaggle.com/code/kcostya/lstm-models-for-multi-step-time-series-forecast#ConvLSTM-Encoder-Decoder-Model-With-Multivariate-Input) <br /> [CNN-LSTM-Based Models for Multiple Parallel Input and Multi-Step Forecast](https://towardsdatascience.com/cnn-lstm-based-models-for-multiple-parallel-input-and-multi-step-forecast-6fe2172f7668) <br /> [使用 LSTM 进行多变量时间序列预测的保姆级教程](https://avoid.overfit.cn/post/1a36216705f2441b80fca567ea61e365)
multiple inputs|multivariate|autoencoder+regression|to predict values next step per sample concerning multivariate variables| [stackoverflow QAs](https://stackoverflow.com/questions/60732647/how-to-have-keras-lstm-make-predictions-for-multiple-time-series-in-a-multivaria)<br />[stackOverflow QAs](https://stackoverflow.com/questions/65144346/feeding-multiple-inputs-to-lstm-for-time-series-forecasting-using-pytorch)
|multiple inputs| multivariate| autoencoder+clustering| to cluster the patterns| [Ｔimes Series clustering and dimensions reduction](https://towardsdatascience.com/time-series-clustering-and-dimensionality-reduction-5b3b4e84f6a3)
|multiple inputs | univariate|  | classification for next moves |[Early Event Detection in Power Lines](https://www.kaggle.com/code/shitalborganve/early-event-detection-in-power-lines/notebook) 
|multiple inputs| univariate| Prophet | anomaly detections|[real time anomaly detections using prophet](https://www.kaggle.com/code/vigneshvdas/real-time-anomaly-detection-using-prophet) <br /> [peak identification](https://www.kaggle.com/code/johnowhitaker/peak-identification/notebook) <br /> [pytoch anomaly detections](https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/)



