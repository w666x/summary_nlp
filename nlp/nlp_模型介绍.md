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
