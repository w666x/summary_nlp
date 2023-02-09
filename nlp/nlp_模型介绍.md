### 模型对比


#### rnn vs cnn

- 差异点对比
    - （1）CNN代表卷积神经网络；RNN代表递归神经网络。
    - （2）CNN适合做图像和视频处理；RNN是文本和语音分析的理想选择。
    - （3）CNN网格采用固定大小的输入并且生成固定大小的输出；**RNN可以处理处理任意长度的输入或者输出长度**。

<!-- #region -->
#### LSTM vs GRU
- 差异点对比
    - （1）LSTM和GRU的性能在很多任务上不分伯仲；
    - （2）GRU参数更少，因此更容易收敛，但是在大数据集的情况下，LSTM性能表现更好；
    - （3）从结构上说，GRU只有两个门，LSTM有三个门，GRU直接将hidden state传给下一个单元，而LSTM则用momery cell把hidden state包装起来。
    

- 模型示例demo
![LSTM结构图](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/LSTM结构图_带维度.jpg)
![GRU更新门](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/GRU结构图.jpg)

<!-- #endregion -->

<!-- #region -->
#### word2vec对比

- 主要有Skip-Gram和CBOW两种，
    - **从直观上讲，Skip-Gram是给定input word预测上下文，而CBOW是给定上下文，来预测input word**。
    - 总体上说，skip-gram的训练时间更长，对于一些出现频率不高的词，在CBOW中的学习效果就不如Skip-Gram，skip-gram准确率更高。


- CBOW模型中input是context（周围词）而output是中心词，
    - 训练过程中其实是在从output的loss学习周围词的信息也就是embedding，但是在中间层是average的，一共预测V(vocab size)次就够了。
- skipgram是用中心词预测周围词，
    - 预测的时候是一对word pair，等于对每一个中心词都有K个词作为output，
    - 对于一个词的预测有K次，所以能够更有效的从context中学习信息，但是总共预测K*V词。
<!-- #endregion -->
