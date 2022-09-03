# **時間序列** 
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
* LSTM batch 
	>[LSTM batch learning2](https://stackoverflow.com/questions/65144346/feeding-multiple-inputs-to-lstm-for-time-series-forecasting-using-pytorch) <br/>
	>[batch learning2](https://www.reddit.com/r/MLQuestions/comments/rn4j5p/how_to_train_one_lstm_model_with_independent/) <br/>
	>[Beijing PM2.5](https://blog.csdn.net/weixin_42608414/article/details/99886972) <br/>
	>[Stateful and stateless 1](https://fairyonice.github.io/Stateful-LSTM-model-training-in-Keras.html) <br>
	>[stateful and stateless 2](https://zhuanlan.zhihu.com/p/34495801)

---

* ## 因果關係檢定

  - The Granger Causality test is used to determine whether or not one time series is useful for forecasting another.


***

## **Models for Times Series**

| samples data | variables | method| Aim |examples
| :--:| :--: | :--:| :-------:|  :-----:|
| multiple inputs / parellel inputs | univariate | autoencoder + MLP regression |predict future value per sample individually |-[store sales predictions](https://www.kaggle.com/code/dimitreoliveira/time-series-forecasting-with-lstm-autoencoders/notebook)   <br />[Web Traffic Time Series Forecasting](https://www.kaggle.com/code/ganeshhalpatrao/web-traffic-time-series-forecasting)
| One input| multiivariate | CNN.autoencoder + keras.LSTM| predict future value with given multiple variables into concerns| [LSTM Models for multi-step time series forcast](https://www.kaggle.com/code/kcostya/lstm-models-for-multi-step-time-series-forecast#ConvLSTM-Encoder-Decoder-Model-With-Multivariate-Input) <br /> [CNN-LSTM-Based Models for Multiple Parallel Input and Multi-Step Forecast](https://towardsdatascience.com/cnn-lstm-based-models-for-multiple-parallel-input-and-multi-step-forecast-6fe2172f7668) <br /> [使用 LSTM 进行多变量时间序列预测的保姆级教程](https://avoid.overfit.cn/post/1a36216705f2441b80fca567ea61e365)
｜multiple inputs|multivariate|autoencoder+regression|to predict values next step per sample concerning multivariate variables| [stackoverflow QAs](https://stackoverflow.com/questions/60732647/how-to-have-keras-lstm-make-predictions-for-multiple-time-series-in-a-multivaria)<br />[stackOverflow QAs](https://stackoverflow.com/questions/65144346/feeding-multiple-inputs-to-lstm-for-time-series-forecasting-using-pytorch)
|multiple inputs| multivariate| autoencoder+clustering| to cluster the patterns| [Ｔimes Series clustering and dimensions reduction](https://towardsdatascience.com/time-series-clustering-and-dimensionality-reduction-5b3b4e84f6a3)
|multiple inputs | univariate|  | classification for next moves |[Early Event Detection in Power Lines](https://www.kaggle.com/code/shitalborganve/early-event-detection-in-power-lines/notebook) 
|multiple inputs| univariate| Prophet | anomaly detections|[real time anomaly detections using prophet](https://www.kaggle.com/code/vigneshvdas/real-time-anomaly-detection-using-prophet) <br /> [peak identification](https://www.kaggle.com/code/johnowhitaker/peak-identification/notebook) <br /> [pytoch anomaly detections](https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/)



