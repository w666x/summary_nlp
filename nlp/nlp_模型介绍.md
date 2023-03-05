### 模型对比


#### rnn vs cnn

- 差异点对比
    - （1）CNN代表卷积神经网络；RNN代表递归神经网络。
    - （2）CNN适合做图像和视频处理；RNN是文本和语音分析的理想选择。
    - （3）CNN网格采用固定大小的输入并且生成固定大小的输出；**RNN可以处理处理任意长度的输入或者输出长度**。

<!-- #region -->
#### LSTM vs GRU
- 差异点对比，详细可参考[循环神经网络](https://github.com/w666x/blog_items/blob/main/04_nlp/04_循环神经网络.md)
    - （1）LSTM和GRU的性能在很多任务上不分伯仲；
    - （2）GRU参数更少，因此更容易收敛，但是在大数据集的情况下，LSTM性能表现更好；
    - （3）从结构上说，GRU只有两个门，LSTM有三个门，GRU直接将hidden state传给下一个单元，而LSTM则用momery cell把hidden state包装起来。
    

- 模型示例demo
![LSTM结构图](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/LSTM结构图_带维度.jpg)
![GRU更新门](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/GRU结构图.jpg)

<!-- #endregion -->

<!-- #region -->
#### HMM vs CRF
- 区别，详细内容可参考[EM&HMM和CRF模型](https://github.com/w666x/blog_items/blob/main/04_nlp/EM&HMM&CRF模型.md)
    - 1.HMM是生成模型，CRF是判别模型；
    - 2.CRF利用的是马尔科夫随机场（无向图），而HMM的基础是贝叶斯网络（有向图）；
    - 3.在概率计算问题、学习问题和预测问题上有差异；
    - 4.HMM求解的是局部最优解，CRF求解的是全局最优解。
    
    
- 马尔科夫模型的问题
    - 因为HMM模型其实它简化了很多问题，做了某些很强的假设，如齐次马尔可夫性假设和观测独立性假设，
        - 做了假设的好处是，简化求解的难度，
        - 坏处是对真实情况的建模能力变弱了。
    - 比如，在序列标注问题中，隐状态（标注）不仅和单个观测状态相关，还和观察序列的长度、上下文等信息相关。
    - 比如，词性标注问题中，一个词被标注为动词还是名词，不仅与它本身以及它前一个词的标注有关，还依赖于上下文中的其他词。
    - 如下图所示，可以使用最大熵马尔科夫模型进行优化。
    

- CRF简述
    - 首先 X,Y 是随机变量，P(Y/X)是给定 X 条件下 Y 的条件概率分布
    - 如果 Y 满足马尔可夫满足马尔科夫性，及不相邻则条件独立，则条件概率分布 P(Y|X)为条件随机场 CRF
    
    
![HMM改进](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/HMM改进.jpg)
<!-- #endregion -->

#### bert vs elmo vs cnn

<!-- #region -->
- ELMo VS Bert
    - 详细对比，可参考[bert介绍](https://github.com/w666x/blog_items/blob/main/04_nlp/bert介绍.md)
    - ELMo模型通过构建词嵌入动态调整的双向神经网络，能够提取上下文敏感特征，输出体现上下文语义的预训练词向量，较好地**解决了一词多义现象**。
        - **ELMo使用了LSTM而不是Transformer来作为特征抽取器，Transformer提取特征的能力是要远强于LSTM的；**
        - 训练时间长，这也是RNN的本质导致的；
    - BERT相比于ELMo模型进一步拓宽了词向量的泛化能力，能够充分学习字符级、词级、句子级甚至句间关系特征，增强字向量的语义表示 ，因此表现出优于过往方法的卓越性能
    - BERT模型采用了多层的双向Transformer编码器结构，同时受到左右语境的制约，相比ELMo模型中独立训练从左到右和从右到左的LSTM连接，能够更好地包含丰富的上下文语义信息。
    

- BERT vs RNN
    - 传统的RNN和CNN在处理NLP任务时存在着 一定缺陷：RNN的循环式网络结构没有并行化，训练慢；CNN先天的卷积操作不是很适合序列化的文本。
    - Transformer模型是文本序列网络的一种新架构，基于self-attention机制，任意单元都会交互，没有长度限制问题，能够更好捕获长距离上下文语义特征。

    
    

- ELMo vs GPT
    - GPT的预训练过程，其实和 ELMO 是类似的，但是：
    - 特征抽取器不是用的 RNN，而是用的 Transformer，
    - ELMO使用上下文对单词进行预测，而 GPT 则只采用这个单词的上文来进行预测，而抛开了下文。
    
    
- GPT vs Bert
    - GPT 预训练时利用上文预测下一个单词，ELMO和BERT是根据上下文预测单词
    - GPT 更加适合用于文本生成的任务，因为文本生成通常都是基于当前已有的信息，生成下一个单词。
<!-- #endregion -->

<!-- #region -->
#### bi-lstm + CRF vs lstm
- bi-lstm vs lstm
    - 与LSTM相比，双向长期短时记忆网络（Bi-LSTM）对每个句子分别采用顺序（从第一个词开 始，从左往右递归）和逆序 （从最后一个词开始， 从右向左递归） 计算得到两套不同的隐层表示，然后通过向量拼接得到最终的隐层表示。
    - 因此，Bi-LSTM能够更好地捕捉双向的语义依赖关系，有效地学习并掌握上下文语义共现信息，从而提升命名实体识别的性能。
    
    
    
- 为啥要加上CRF
    - 然而，BiLSTM不考虑标签之间的相关性， 而CRF的一个独特优势是能够通过考虑相邻标签的关系获得一个全局最优的标记序列。
    - 将CRF与BiLSTM神经网络相结合，对BiLSTM的输出进行处理，可以获得最佳的术语标记结果。
<!-- #endregion -->

<!-- #region -->
#### 生成模型 vs 判别模型
- 生成模型既可以做数据生成，也可以做分类判别；而判别模型只能做判别任务
    - 详细内容，可参考[EM&HMM&CRF模型](https://github.com/w666x/blog_items/blob/main/04_nlp/EM&HMM&CRF模型.md)

| 模型分类 | 功能 | 常见模型 | 目标函数 | 特点
|:- |:- |:- |:-|:-
| 生成模型 | 文本**生成**、判别 | HMM、朴素贝叶斯、GPT | 对联合分布进行建模，最终学习到的是关于x/y的分布 | **需要学习样本的分布信息** <br>学习得到联合概率分布P(x,y),即特征x和标记y共同出现的概率
| 判别模型 | 分类**判别** | CRF、逻辑回归、感知机、决策树 | 对条件分布进行建模，最终学习到的是关于y的分布 | 只需要学习不同分类的样本的差异就够了 <br> **学习得到条件概率分布P(y\|x)**,即在特征x出现的情况下标记y出现的概率。



1. HMM模型
    - 由隐藏的马尔可夫链随机生成观测序列，是生成模型。
    - HMM是关于时序的概率模型，描述由一个隐藏的马尔可夫链随机生成不可观测的状态随机序列，再由各个状态生成一个观测而产生观测随机序列的过程。
    - 包含三要素： $初始状态概率向量\pi$，状态转移概率矩阵A，观测概率矩阵B。
    
    
2. 判别模型-最大熵模型
    - 原理：概率模型中，熵最大的模型是最好的模型，可以使用拉格朗日函数求解对偶问题解决。


3. 判别模型-支持向量机(SVM)
    - SVM分为线性可分支持向量机(硬间隔最大化)、线性支持向量机(软间隔最大化)、非线性支持向量机(核函数)三种。
    - 目的是最大化间隔，这是和感知机最大的区别。


4. 判别模型-boosting方法(AdaBoost等)
    - 通过改变训练样本的权重，训练多个分类器，将分类器进行线性组合，提升分类性能。
    - AdaBoost采用加权多数表决的方法。


5. 判别模型-条件随机场(conditional random field, CRF)
    - 给定一组输入随机变量条件下另一组输出随机变量的条件概率分布模型，
    - 其特点是假设输出随机变量构成马尔可夫随机场。可应用于标注问题。
<!-- #endregion -->

<!-- #region -->
#### 词向量

- word2vec是一个把词语转化为对应向量的形式。word2vec中建模并不是最终的目的，其目的是获取建模的参数。


##### 概览
- 各词向量方法的特点，详细词向量内容可参考[词向量介绍](https://github.com/w666x/blog_items/blob/main/04_nlp/00_word2vec.md)


|方法名称 | 特点 | 说明 | 优点 | 缺点
|:- |:- |:- |:- |:-
| one-hot | 0/1/0/1 | 独热编码，即用一个很长的向量来表示一个词，词的长度为词典D的大小N，**向量的分量只有1个1，其余的都是0.** | 简单 | 维度灾难、无法刻画词与词之间的相似性
| [MF](#MF) | 依赖共现矩阵的构造 | 全称是word embedding by Matrix Factorization，矩阵分解算法 | 其得到的词向量更好的包含整体信息<br>语义相近的词在向量空间相近，甚至可以一定程度反映word间的线性关系<br>在一定程度上缓解了one-hot向量相似度为0的问题 | 很多词没有出现，导致矩阵极其稀疏，因此需要对词频做额外处理来达到好的矩阵分解效果；<br> 矩阵非常大，维度太高。没有解决数据系数和维度灾难的问题 <br> 需要手动去掉停用词
| LSA | 矩阵分解 | 主题模型，潜在语义分析 | 利用全局语料特征 | SVD矩阵分解求解计算复杂度大
| [skip-gram](#跳词) | 即**用中间的单词，来预测其周围的单词** | 对生僻字有利，总体效果相对cbow较好 | 模型训练速度相对较慢；<br> 训练词向量同上下文无关
| [cbow](#cbow方法) | 根据给定**上下文的词 $w_{input}$，预测中间目标词出现的概率 $w_t$** | 全称为Continuous Bag-of-Word Model，**词袋模型** | CBOW模型训练速度快 | 训练词向量同上下文无关
| [fastText](#fasttext) | 在skip-gram模型的基础上，将**中心词的向量表示成了n-gram单词的子向量之和**<br>fasttext可以有效**解决OOV**（out of vocabulary）的情况<br>一般情况下，使用fastText进行文本分类的同时也会产生词的embedding，即embedding是fastText分类的产物。 | 结构同skip-gram相似  | 效率高 | 基于局部语料
| [glove](#Glove) | 需要提前训练的，**词向量是静态的**，预测时上下文无关 | 等价于MF+Skip-gram | **基于全局语料**，结合了LSA（MF）和word2vec（skip-gram）的优点
| elmo/GPT/bert |假设向下文语境的词有相似含义；词向量为副产物，<br>**词向量是动态的**，其和上下文是相关的 | 预训练模型 | 动态特征，可以解决**一词多义问题** | 效率较低
<!-- #endregion -->

##### 常见问题

<!-- #region -->
1. word2vec vs tf-idf
    - Word2vec是稠密的向量，而 tf-idf 则是稀疏的向量；
    - Word2vec的向量维度一般远比 tf-idf 的向量维度小得多，故而在计算时更快；
    - Word2vec的向量可以表达语义信息，但是 tf-idf 的向量不可以；
    - Word2vec可以通过计算余弦相似度来得出两个向量的相似度，但是 tf-idf 不可以。
    - tf-idf只是基于词频信息、无上下文关联、无顺序信息。
    
    
2. Word2vec vs NNLM
    - NNLM：（Nature Neural Language Model）是神经网络语言模型，使用前n-1个单词预测第n个单词;
    - word2vec：使用第n-1个单词预测第n个单词的神经网络模型。但是**word2vec更专注于它的中间产物词向量，所以在计算上做了大量的优化**。优化如下：
        - 对输入的词向量直接按列求和，再按列求平均。这样的话，输入的多个词向量就变成了一个词向量。
        - 采用分层的 softmax(hierarchical softmax)，实质上是一棵哈夫曼树。
        - 采用负采样，从所有的单词中采样出指定数量的单词，而不需要使用全部的单词
        
        
        
3. Word2vec训练trick，词向量维度大与小有什么影响，还有其他参数？
    - **词向量维度代表了词语的特征，特征越多能够更准确的将词与词区分**，就好像一个人特征越多越容易与他人区分开来。
    - 但是在实际应用中维度太多训练出来的模型会越大，虽然维度越多能够更好区分，但是词与词之间的关系也就会被淡化，这与我们训练词向量的目的是相反的，我们训练词向量是希望能够通过统计来找出词与词之间的联系，
        - 维度太高了会淡化词之间的关系，但是维度太低了又不能将词区分，
        - 所以词向量的维度选择依赖于你的实际应用场景，这样才能继续后面的工作。
    - 一般说来200-400维是比较常见的。windows窗口默认参数是5。
    
    
4. 【tf-idf计算】在包含 N 个文档的语料库中，随机选择一个文档。该文件总共包含 T 个词，词条「数据」出现 K 次。如果词条「数据」出现在文件总数的数量接近三分之一，则 TF（词频）和 IDF（逆文档频率）的乘积的正确值是多少？
    - tf-idf计算公式是， $TF-IDF = TF * IDF$，其中，$TF = \frac{文章中出现的次数}{文章总词数或者最高词频数}$，逆文档频率：$IDF = \log\frac{语料库的总数}{包含该词的文档数+1}$ 
    - 则，上述问题答案为， $\frac{K}{T} * \log3$
    
    
5. 【Glove】解释 GolVe 的损失函数？ 
    - 其实，一句话解释就是想构造一个向量表征方式，使得向量的点击和共现矩阵中的对应关 系一致。
    - 因为共现矩阵中的对应关系证明了，存在 i，k，j 三个不同的文本，如果 i和k 相关，j 和 k 相关，那么 p(i,j)=p(j,k)近似于1，其他情况都过大和过小。 
    - 如何处理未出现词？
        - 按照词性进行已知词替换，[unknow-n],[unknow-a],[unknow-v]...，然后再进行训练。
        - 实际去用的时候，判断词性后直接使用对应的 unknown-?向量替代 
        
        
6. 【Glove】为什么 GolVe 会用的相对比W2V 少？ 
    - GloVe 算法本身使用了全局信息，自然**内存费的也就多一些**，共现矩阵，NXN 的，N为词袋量，
    - W2V 的工程实现结果相对来说支持的更多，比如 most_similarty 等功能 
    
    
7. 【embedding】怎么衡量学到的 embedding 的好坏 
    - 从item2vec得到的词向量中随机抽出一部分进行人工判别可靠性。即人工判断各维度item与标签item的相关程度，判断是否合理，序列是否相关 
    - 对item2vec得到的词向量进行聚类或者可视化
    
    
8. LDA和 Word2Vec 区别？LDA 和 Doc2Vec 区别 
    - LDA 比较是doc，word2vec 是词 
    - LDA 是生成的每篇文章对 k 个主题对概率分布，Word2Vec 生成的是每个词的特征表示
    - LDA 的文章之间的联系是主题，Word2Vec 的词之间的联系是词本身的信息
    - LDA 依赖的是doc和word共现得到的结果，Word2Vec依赖的是文本上下文得到的结果
<!-- #endregion -->

<!-- #region -->
##### 词向量方法介绍
- 主要包括，


1. 矩阵分解 <b id="MF"></b> ，**共现矩阵**的生成步骤：
    - 首先构建一个空矩阵，大小为V × V，即词库表×词库表，值全为0。
    - 确定一个滑动窗口的大小(例如取半径为5)
    - 从语料库的第一个单词开始，以1的步长滑动该窗口，按照语料库的顺序开始的，所以中心词为到达的那个单词即i。
    - 上下文环境是指在滑动窗口中并在中心单词i两边的单词(这里应有2*5-1个单词)。
        - 若窗口左右无单词，一般出现在语料库的首尾，则空着，不需要统计。
        - **在窗口内，统计上下文环境中单词j出现的次数，并将该值累计到矩阵(i,j)位置上。**
    - 不断滑动窗口进行统计即可得到共现矩阵。
    
    
![词向量-矩阵分解](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/词向量-矩阵分解.jpg)


2. 算法介绍-SkipGram <b id="跳词"></b> 
    - 即**用中间的单词，来预测其周围的单词**  
    - 假设一组词序列为即为text，为 $w_1,w_2,\cdots,w_T$，则SkipGram算法的目标为最大化似然函数L： $\displaystyle L=\sum_{w\in text}\sum_{c \in context(w)}logp(c|w)，$
        - 其中text为当前序列中的所有词；w表示词库中的任意词；
        - context(w)表示w作为中性词是其上下文（batch size）的词
    - 另有，  $\displaystyle p(w_O|w_I) = \frac{exp({v\text{'}_{w_O}}^Tv_{w_I})}{\sum_{w=1}^Wexp({v_{w}}^Tv_{w_I})}$,其中， $v_{wi}和v_{wo}$是输入输出向量；**W为词汇表中单词的个数**
    - 涉及的参数
        - window size：即根据中间的词，要预测周围多少词
    - 最后的结果，可以理解为是在多分类
        - 向量之间越相似，点乘结果越大。而通过softmax后，得到的概率值也就越大。
       
    
    
![skipgram计算实例](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/skipgram计算实例-基础.jpg)


3. 算法介绍-cbow方法 <b id="cbow方法"></b> 
    - cbow模型的全称为Continuous Bag-of-Word Model，词袋模型。
    - 该模型的作用是根据给定**上下文的词 $w_{input}$，预测中间目标词出现的概率 $w_t$**，对应的数学表示为 $p(w_t|w_{input})$
    - 连续词袋模型模型
        - 首先就是定义损失函数，这个损失函数就是给定输入上下文的输出单词的条件概率，一般都是取对数，如下:
            $$E=−logp(w_O|w_I)= -log(\frac{exp({v_{w_O}}^Th)}{\sum_{j=1}^Vexp({v_{wj}}^Th)}) =−v_{wo}^T\cdot h−log\sum_{j′=1}^Vexp(v^T_{wj′}\cdot h)$$
        - 接下来就是对上面的概率求导，具体推导过程可以去看BP算法，我们得到输出权重矩阵W′的更新规则：
            $$w′^{new}=w′^{old}_{ij}−\eta\cdot(y_j−t_j)\cdot h_i$$
        - 同理权重W的更新规则如下：
            $$w^{new}=w^{old}_{ij}−\eta \cdot\frac{1}{C}\cdot EH$$
    
![cbow计算实例](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/cbow计算实例.jpg)



4. 算法介绍-fasttext <b id="fasttext"></b> 
    - fastText模型架构和word2vec中的skip-gram很相似，不同之处是**fastText词典规模更大，模型参数更多**
        - 在skip-gram模型的基础上，将**中心词的向量表示成了n-gram单词的子向量之和**
        - fasttext可以有效**解决OOV**（out of vocabulary）的情况
        - 一般情况下，使用fastText进行文本分类的同时也会产生词的embedding，即embedding是fastText分类的产物。
    - fastText的模型也是三层架构：
        - 输入层、 隐藏层、输出层（Hierarchical Softmax）。
        - fastText的**输入是多个单词及其n-gram特征**，这些特征用来表示单个文档，将整个文本作为特征去预测文本对应的类别。
    - 模型特点
        - （1）模型总体只有三层，结构简单；
        - （2）文本表示的向量简单相加平均；
        - （3）在输出时，fastText采用了分层Softmax，大大降低了模型训练时间；


5. 算法介绍-Glove <b id="Glove"></b> 
    - GloVe的全称是：Global Vectors for Word Representation
    - 任意词的**中心词向量和上下文词的向量，在Glove模型中是等价的**
    - 特点
        - 使用了共现次数（详细可见MF矩阵分解构建词向量部分）
        - 使用了上下文窗口大小（详见skip-gram部分）
        - 使用了**平方损失** $J = \sum_{i,j=1}^Vh(X_{ji})(w_i^Tw_j + b_i + \overline b_j - logX_{ij})^2$
        - $其中，b_i为中心词偏差项； \overline b_j为上下文词偏差项$
        - $损失项的权重h(X_{ji})为值域在[0,1]上的单调递增函数；且词频过高，权重不宜过大$,
       
$$h(x) = \begin{cases} (\frac{x}{c})^{\alpha} & x \le c ,\text{比如c = 100, $\alpha$ = 0.75} \\ 1 & x> c \end{cases}$$

![词向量-glove实例](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/glove实例.jpg)


##### skip gram vs cbow


- 主要有Skip-Gram和CBOW两种，
    - **从直观上讲，Skip-Gram是给定input word预测上下文，而CBOW是给定上下文，来预测input word**。
    - 总体上说，skip-gram的训练时间更长，对于一些出现频率不高的词，在CBOW中的学习效果就不如Skip-Gram，skip-gram准确率更高。


- CBOW模型中input是context（周围词）而output是中心词，
    - 训练过程中其实是在从output的loss学习周围词的信息也就是embedding，但是在中间层是average的，一共预测V(vocab size)次就够了。
- skipgram是用中心词预测周围词，
    - 预测的时候是一对word pair，等于对每一个中心词都有K个词作为output，
    - 对于一个词的预测有K次，所以能够更有效的从context中学习信息，但是总共预测K*V词。
    
    
- 为啥子一般情况下，skip-gram的效果都比cbow的要好嘞？
    - 数据量的关系呀（cbow是周围词预测中间词；cbow是中间词分别预测周围词）导致，**同样的文本，skip-gram的训练的数据量会比cbow的多啦**。
    - smoth effect，cbow的训练过程中，是对周围词的one-hot表征进行求平均后来预测中心值的，这种平滑处理对出现次数小的词不利
    - **skip-gram训练难度比cbow的大**
<!-- #endregion -->

<!-- #region -->
##### word2vec优化部分

- 为啥word2vec中要使用负采样
    - 使用霍夫曼树来代替传统的神经网络，可以提高模型训练的效率。**但是如果我们的训练样本里的中心词w是一个很生僻的词，那么就得在霍夫曼树中辛苦的向下走很久了**
    - 而负采样一种概率采样的方式，不需要使用全部样本来更新模型参数了，只需要负采样规模的样本即可
    

| 方法 | 特点 | 适用场景 | 复杂度
|:- |:- |:- |:- 
| [Hierarchical softmax](#层次softmax) | 层次softmax，使用了二叉树 [Huffman树](#test1) ，根据根节点到叶节点的路径来构造损失函数 | 在低频词上表现更好| 训练中每一步的梯度计算量与词典大小的对数相关
| [负采样](#负采样) | 通过考虑同时包含正负样本的相互独立事件来构造损失函数 <br> 具体方法为：根据词频进行采样，也就是**词频越大的词被采到的概率也越大**。 | 在高频词和较低维度的向量上表现好 | 训练中每一步的梯度计算量与负采样的个数K呈线性关系
| sub sampling |每个词都有一定的概率被丢弃，词频越高丢弃概率越大 | 样本不均衡
<!-- #endregion -->

<!-- #region -->
- 层次softmax <b id="层次softmax"></b> 
    - 可以优化计算量，使用二叉树来代替word2vec中隐藏层→输出层的映射结构
    - **使用binary tree来表示输出层**，
        - w个词分别表示叶子节点（输出层）；每个节点表示其子节点的相对路径
        - 用n(w,j)来表示从根节点到w词的这条路径上的第j个节点（隐藏层）
        - 用sigmoid函数来判别节点向左子树、右子树转移的概率 $P(+) = \sigma(x_w^T\theta) = \frac{1}{1+e^{-x_w^T\theta}}$
    - 分层softmax选择不同的树形结构，对性能有很大的影响。一般情况，我们会选择Huffman树
    - huffman树的应用
        - 对于词典D中的任意词w，Huffman树中必存在一条从根节点到词w对应节点的路径 $p^w(且这条路径是唯一的)$
        - 路径 $p^w上存在l^w-1个分支$，将每个分支都看成是一个二分类，每一次分类就产生了一个概率，将这些概率乘起来，就得到了所需要的 $p(w|context(w))$啦
        - **每次，合并最小的两棵树**
        - 可以简单且高效的加速训练
        - 高频的词靠近树根，这样高频词只需要更少的时间就可以找到。
    - 优点：计算量变少；由于是二叉树，在输出层不需要计算W个节点，只需要计算log(w)个即可
    - 缺点：如果训练样本中的词比较生僻，则耗时很久
    
    
    
- 层次 softmax 流程 
    - 首先构建哈夫曼树，即以词频作为 n 个词的节点权重，不断将最小权重的节点进行合并，最终形成一棵树，权重越大的叶子结点越靠近根节点，权重越小的叶子结点离根节点越远。
    - 然后进行哈夫曼编码，即对于除根节点外的节点，左子树编码为1，右子树编码为0。
    - 最后采用二元逻辑回归方法，沿着左子树走就是负类，沿着右子树走就是正类，从训练样本中学习逻辑回归的模型参数。
        - 优点：计算量由V（单词总数）减小为 log2V；高频词靠近根节点，所需步数小，低频词远离根节点。
        - 缺点：如果我们的训练样本里的中心词w是一个很生僻的词，那么就得在霍夫曼树中辛苦的向下走很久了。然后，负采样就来啦

![词向量-层次softmax实例](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/skip-gram层次softmax.jpg)
<!-- #endregion -->

<!-- #region -->
- Huffman编码 <b id="test1"></b> 
    - 霍夫曼编码使用**变长编码表**对源符号（如文件中的一个字母）进行编码，
    - 其中**变长编码表是通过一种评估来源符号出现机率的方法得到的，出现机率高的字母使用较短的编码**，反之出现机率低的则使用较长的编码，
    - 这便使编码之后的字符串的平均长度、期望值降低，从而达到无损压缩数据的目的。
    - Huffman编码使得每一个字符的编码都与另一个字符编码的前一部分不同，不会出现解码的问题：**任何字符的编码都不是其它字符的前缀。**
    - 构造方式：**每次把权值最小的两棵树合并**
    

- 最优二叉树
    - 设二叉树有n个叶子结点，每个叶子结点带有权值 $W_k$，从根结点到每个叶子结点的长度为 $l_k$，则**每个叶子结点的带权路径长度之和（WPL）最小的时候**，便称为最优二叉树或者哈夫曼树。
   
$$WPL = \sum W_kl_k$$
    
- 特点
    - 词频越大的词离根节点的越紧
    - 若叶子节点的个数为n，则构造的Huffman树的新增节点个数为n-1

![词向量-Huffman树实例](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/Huffman树实例.jpg)
<!-- #endregion -->

<!-- #region -->
- sub sampling  
    - 为解决高频词、低频次之间的不平衡问题，通过二次采样平衡数据，可以更好的学习到低频次
    - 每个词都有一定的概率被丢弃，其丢弃概率定义为：

$$P(w_i) = 1 - \sqrt{\frac{1}{f(w_i)}}\quad \text{其中，f(w)为每个词的词频}$$

    
- 负采样 <b id="负采样"></b> 
    - 负采样的核心思想是：**利用负采样后的输出分布来模拟真实的输出分布**
    - 它修改了原来的目标函数。
    - 1）给定中心词$w_c一个上下文窗口，我们把上下文词w_o$出现在该上下文窗口看作是一个事件，且该事件的概率定义为： $$P(D=1|w_c,w_o) = \sigma(u_o^Tv_c)$$
    - 2）给定一个长度为T的文本序列，设时间步t的词为 $w^{(t)}且背景窗口的大小为m$；若上下文词出现 $w_o出现在中心词w_c$的窗口为事件P，我们根据分布P(w)采样k个未出现在该上下文窗口中的词，称为**噪声词**
    - 3）设噪声词 $w_k(k=1,\cdots,K)$不出现在中心词的上下文窗口为 $事件N_k；假设同时含有正样本和负样本的事件P、N_1、\cdots、N_K相互独立$，则我们的目标函数即可以改写为：
$$\prod_{t=1}^T\prod_{-m\le j\le m, j\neq 0}P(D=1|w^{(t)}, w^{(t+j)})\prod_{k=1,w_k\text{~}P(w)}^KP(D=0|w^{(t)},w_k)$$

- **NEG负采样(negative sampling)不是多分类而是二分类问题**
    - 给定两个词，其中一个词作为content，来判断另一个词是不是target
    - 如果是，则输出1；否则输出0
    - NCE(noise contrastive estimation)噪声对比估计
        - 假设一个好的模型可以通过逻辑回归来区分正常数据和噪声
- 训练数据的构造为，一组正取样加上n组负取样
    - 正取样：上下文词同中心词是对应的
    - 负取样：上下文词同中心词并不是对应的
    - $\color{blue}{如何选择负样本嘞}$
        - 一般小数据的话n选择5-20，大数据集的话n选择2-5
        - 根据文本中单词的词频取样；
        - 均匀分布的随机采样
        - 加权采样，其中权重为 $P(w_i) = \frac{f(w_i)^{3/4}}{\sum_{j=1}^nf(w_j)^{3/4}}，其中n为词库大小$
- 词向量和字典中的每一个词都会进行一次二分类，来判断这个词是否是content的target
    - 每次迭代只涉及到1+n个单元(其中1为正例；n为负例，比如n=20啦)

- NCE(noise contrastive estimation)噪声对比估计
    - **假设一个好的模型可以通过逻辑回归来区分正常数据和噪声**
   
![词向量-skipgram负采样](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/skip-gram负采样.jpg)


   
<!-- #endregion -->

### 模型定义

<!-- #region -->
#### CNN模型
- CNN详细内容可参考，[卷积神经网络](https://github.com/w666x/blog_items/blob/main/04_nlp/03_卷积神经网络.md)
- CNN是一种前馈神经网络，通常包含5层，输入层，卷积层，激活层，池化层，全连接FC层，其中核心部分是卷积层和池化层。
    - 优点：**共享卷积核**，对高维数据处理无压力；无需手动选取特征。局部连接，相对于全连接而言，减少了参数量哈。
    - 缺点：需要调参；需要大量样本。
    
    
| 网络层 | 结构 | 定义 | 特点
|:- |:- |:- |:-
| 输入层 | 二维输入数组
| 卷积层 |卷积层的输出形状，由**输入形状**和**卷积核窗口形状**决定  | 通常在卷积层中使用更加直观的**互相关运算** <br> 二维卷积层中，我们是将和一个二维核数组通过互相关运算输出一个二维数组，其中二维核数组也称为**卷积核或者过滤器filter** | 卷积层的超参数：填充和步幅 <br> 填充：通常指的是在输入高和宽的两侧填充元素（通常为0） <br> 步幅：将**每次滑动的行数和列数称为步幅**
| 激活层 | sigmoid激活函数等 | | 提取非线性信息啦
| 池化层 | 比如最大池化Max-pooling，取整个区域的最大值作为特征 | 取整个区域的值作对应运算 | 最为常见，在自然语言处理中常用于分类问题，希望观察到的特征是强特征，以便可以区分出是哪一个类别
| 全连接层 | 线性全连接层


- 卷积核的特点
    - 1、大卷积核用多个小卷积核代替；
    - 2、单一尺寸的卷积核用多尺寸卷积核代替；
    - 3、固定形状卷积核趋于使用可变形卷积核；
    - 4、使用1*1卷积核；
    - **5、通过小卷积核减少通道个数（1* 1），通过大卷积核来提取特征（5* 5）**
    
![CNN在文本中的应用](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/CNN在文本中的应用.png)
![多通道卷积示例](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/多通道卷积.jpg)
![TextCNN卷积示例](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/TEXTCNN卷积示例.jpg)
<!-- #endregion -->

<!-- #region -->
#### HMM算法

- 常见的3个问题

| 问题分类 | 说明 | demo | 求解方式
|:- |:- |:- |:-
| **概率计算问题** | 给定模型 $\lambda = (A,B,\pi)和观测序列O=(o_1,o_2,\cdots,o_T)$，计算该观测序列出现的概率 | 模型 $\lambda = (A,B,\pi)$已知，观测序列为 $Q = 白 \rightarrow 黑 \rightarrow 白 \rightarrow 白 \rightarrow 黑 $ | 需要求解 $P(Q\|\lambda)$，即观测序列Q发生的概率，可通过**前向-后向算法**求解
| **学习问题** | 已知观测序列 $O=(o_1,o_2,\cdots,o_T)$，估计模型 $\lambda = (A,B,\pi)$的参数，使得在该模型下观测序列概率 $P(O\|\lambda)$最大| demo为：根据观测序列 $Q = 白 \rightarrow 黑 \rightarrow 白 \rightarrow 白 \rightarrow 黑 $，去寻找模型的一组隐状态参数 $\lambda = (A,B,\pi)$，使得模型在观测序列发生时， $P(Q\|\lambda)$最大， | 使用EM算法求解
| **预测问题**（即解码问题） | 已知模型和观测序列 $O=(o_1,o_2,\cdots,o_T)$，求对给定观测序列条件概率 $P(I\|O)$最大的状态序列 $I=(i_1,i_2,\cdots,i_t)$ | demo为：已知观测序列为 $Q = 白 \rightarrow 黑 \rightarrow 白 \rightarrow 白 \rightarrow 黑 $，当已知模型参数 $\lambda = (A,B,\pi)$后，求出哪一种状态序列发生的可能性最大。 | 抽取什么样的盒子顺序（状态序列），更有可能得到 $Q = 白 \rightarrow 黑 \rightarrow 白 \rightarrow 白 \rightarrow 黑 $的观测结果，**使用维特比算法**、动态规划算法求解


- 马尔科夫过程
    - 假设一个随机过程，其 $t_n时刻的状态x_n，只与t_{n-1}时刻的状态x_{n-1}$相关
    - 一句话去概括，即**当前时刻状态仅与上一时刻状态相关，与其他时刻不相关。【1阶马尔科夫】**
    - 可以从马尔可夫过程图去理解，由于每个状态间是以有向直线连接，也就是当前时刻状态仅与上一时刻状态相关。


- 隐马尔可夫链
    - 生成一个长度为n的序列
        - for i in range(1, n):
            - 基于之前的状态，新生成一个状态的概率为：$P(t_i|t_{i-1})$
            - 基于给定的状态，生成一个观测值$P(w_i|t_i)$
    - **前提假设**
        - 每个位置的状态只和它前一个位置的状态有关（齐次马尔科夫假设）
        - 每一个位置的观测值和当前位置的状态有关（观测独立性假设）



    
- 确定HMM，有如下定义
    - 设Q是所有可能的状态集合，V是所有可能的观测集合； $Q={q_1,q_2,\cdots,q_N}；V={v_1,v_2,\cdots,V_M}$
    - N是可能的状态数，M是可能的观测数
    - I是长度为T的状态序列；O是对应的观测序列； $I=(i_1,i_2,\cdots,i_t)；O=(o_1,o_2,\cdots,o_T)$
    - 状态转移概率矩阵，**模型在各个状态间转换的概率**，记为 $A=[a_{ij}]_{NxN}$，其中，
    $$a_{ij}=P(i_{t+1}=q_j\|i_t=q_i),\quad 1< i,j < N, \text{表示时刻t处于状态 $q_i$的条件下时刻t+1转移到状态 $q_j$的概率}$$
    - 观测概率矩阵,**根据当前状态获取各个观测值的概率**，记为 $B=[b_j(k)]_{NxM}$，其中，
    $$b_j(k) = P(o_t=v_k|i_t=q_j),\quad k=1,2,\cdots,M;j=1,2,\cdots,N,\text{表示时刻t处于状态 $q_j$的条件下生成观测$v_k$的概率}$$
    - 初始状态概率，**模型在初始时刻各状态出现的概率**，记为 $\pi = (\pi_i)$，其中，
    $$\pi_i = P(i_1=q_i), \quad i=1,2,\cdots, N，\text{表示模型的初始状态时刻t=1处于状态 $q_i$的概率}$$

        
        

<!-- #endregion -->

<!-- #region -->
     
- 1）概率问题求解
    - 前向算法，特点是 $\color{red}{联合概率分布}$ ，
        - 定义 $\alpha_t{i} = P(o_1, o_2, \cdots, o_t, i_t = q_i|\lambda), 其中，\alpha_t(i)是o_1, o_2, \cdots, o_t和i_t$的联合概率分布
        - 先计算， $\alpha_1(i), \alpha_2(i), \cdots, \alpha_t(i)$, 每一个时间点的 $\alpha$都使用前一时刻的计算结果， $\alpha_{t+1}(i) = [\sum_{j=1}^N\alpha_t(j)a_{ji}]b_i(o_{t+1}), \quad i=1,2,\cdots,N；t=1,2,\cdots,T-1$
        - 最后，计算到最终的概率计算问题， $P(O|\lambda) = \sum_{i=1}^N\alpha_T(i)$
    - 后向算法，特点是 $\color{red}{条件概率分布}$，
        - 定义 $\beta_t(i) = P(o_{t+1},o_{t+2},\cdots,o_T | i_t=q_i,\lambda), 其中，\beta_t(i)是o_{t+1}, o_{t+2}, \cdots, o_T的关于i_t$的条件概率分布
        - 先计算， $\beta_t(i), \beta_{t-1}(i), \cdots, \beta_1(i)$, 每一个时间点的 $\beta$都使用后一时刻的计算结果， $\beta_{t}(i) = \sum_{j=1}^Na_{ij}b_j(o_{t+1})\beta_{t+1}(j), \quad i=1,2,\cdots,N；t=T-1,T-2,\cdots,1$
        - 最后，计算到最终的概率计算问题， $P(O|\lambda) = \sum_{i=1}^N\pi_ib_i(o_1)\beta_1(i)$
        


![HMM前向计算](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/HMM概率计算-前向.jpg)


![HMM后向计算](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/HMM后向计算.jpg)
       
       
 
<!-- #endregion -->

<!-- #region -->
      
- 2）参数学习问题
    - 问题：已知观测序列，但不知道状态序列，估计隐马尔可夫模型的参数 $P(O|\lambda) = \sum_{I}P(O|I,\lambda)P(I|\lambda)$， 使得在该模型下观测序列概率 $P(O|\lambda)$最大


- 方法：
    - 1）确定完全数据的对数似然函数
        - 观测变量 $O=(o_1,o_2,\cdots,o_T)，隐变量I={i_1,i_2,\cdots, i_T}，完全数据(O,I)={o_1,o_2,\cdots,o_T,i_1,i_2,\cdots,i_T}$
        - 完全数据的对数似然函数是 $logP(O,I|\lambda)$
    - 2）E步，求Q函数
        - 首先定义Q函数
        $$\begin{aligned}Q(\lambda ,\hat\lambda) &= E_{I\text{~}P(I|O,\hat\lambda)}logP(O,I|\lambda) \\
        &= \sum_IP(I|O,\hat\lambda)logP(O,I|\lambda) \\ 
        &= \frac{1}{P(O|\hat\lambda)}\sum_IP(I,O|\hat\lambda)logP(O,I|\lambda)\end{aligned}$$
        - 其中，
        $$P(O,I|\lambda) = \pi_{i_1}b_{i_1}(o_1)a_{i_1i_2}b_{i_2}(o_2)\cdots a_{i_{T-1}i_T}b_{i_T}(o_T)$$
    - 3）M步，极大化Q函数，分别求出$\pi,a,b$的估计值
        - Q函数&约束条件（和为1这些），然后应用拉格朗日乘子法求解
    
    
    
<!-- #endregion -->

- 3）预测问题-维特比算法
    - 最优路径在时刻t通过节点 $i_t^*$，**那么这个从起点到$i_t^*$的部分路径，对所有从起点到$i_t^*$的路径来说，必须是最优的**。

    - $\color{red}{算法3.2}$维特比算法
    - 定义在时刻t状态为i的所有单个路径$(i_1,i_2,\cdots,i_t)$中概率最大值为 $\delta_t(i)$，N为状态总数
        $$\begin{aligned}\delta_t(i) &= \max_{i_1,i_2,\cdots,i_{t-1}}P(i_t=i,i_{t-1},\cdots,i_1,o_t,\cdots,o_1|\lambda), \quad i=1,2,\cdots,N \\
        \delta_{t+1}(i) &= \max_{i_1,i_2,\cdots,i_{t}}P(i_{t+1}=i,i_t,\cdots,i_1,o_{t+1},\cdots,o_1|\lambda),\quad i=1,2,\cdots,N \\ 
        &= \max_{1\le j\le N}[\delta_t(j)a_{ji}]b_i(o_{t+1}),\quad t=1,2,\cdots,N,t=1,2,\cdots,T
        \end{aligned}$$
    - **记录位置定义在时刻t状态为i的所有单个路径 $(i_1,i_2,\cdots,i_t)$中概率最大的路径的第t-1个节点为** ：
    $$\Psi_t(i) = \arg\max_{1\le j\le N}[\delta_{t-1}(j)a_{ji}],\quad i=1,2,\cdots,N$$
    - 输入：模型 $\lambda = (A,B,\pi)和观测O={o_1,o_2,\cdots,o_T}$
    - 输出：最优路径 $I^* = (i_1^*,i_2^*,\cdots,i_T^*)$
    - 步骤：
        - 1）初始化
        $$\begin{aligned}\delta_1(i) &= \pi_ib_i(o_1) = p(o_1, i_1 = i) = p(i_1 = i) * p(o_1 | i_1 = i) \qquad i=1,2,\cdots,N,为可能状态总数 \\
        \Psi_1(i) &= 0 \qquad i=1,2,\cdots, N\end{aligned}$$
        - 2）递推
        $$\begin{aligned}\delta_{t}(i) &= \max_{1\le j\le N}[\delta_{t-1}(j)a_{ji}]b_i(o_{t}) \qquad i=1,2,\cdots,N,为可能状态总数 \\
        \Psi_t(i) &= \arg\max_{1\le j\le N}[\delta_{t-1}(j)a_{ji}]  \qquad i=1,2,\cdots,N,为可能状态总数 \end{aligned}$$
        - 3）终止
            $$\begin{aligned}P^* &= \max_{1\le j\le N}\delta_T{(j)} \\
        i_T^* &= arg\max_{1\le j\le N}[\delta_T(j)]\end{aligned}$$
        - 4）回溯 $i_t^* = \Psi_{t+1}(i_{t+1}^*)$
        
        
- 4）实例，demo

- 以下面例子做说明，当我们计算到第三个汉字“是”后停止，
    - 首先，应用上式算法步骤中的终止过程，计算 $最终概率P^*和最终的最优节点i_T^*$的值，为 $P^*=0.1008、i_3^*=s$
    - 然后，进行回溯，即： $i_2^* = \Psi_3(s) = e \rightarrow i_1^* = \Psi_2(e) = b$
    - 综上，即得到前三个状态序列为：(b, e, s)
- 定义，
    - 定义在时刻t状态为i的所有单个路径 $(i_1,i_2,\cdots,i_t)$中概率最大值为 $\delta_t(i)$
    - 另外，由于对于任意一个时刻的 $\Psi_t(i)其和\delta_{t}(i)$的唯一区别就是，多了1个 $b_i(o_t)$, 而在计算任意1个状态i时，这个值是固定值，所以 $\delta_{t}(i)$最大的状态i，也是 $\Psi_t(i)$最大的状态
    - 比如， $\delta_{2}(e) = 0.252，对应的是\delta_{1}(b)*a_{be}*b_e(o_2) = 0.252，则\Psi_2(e) = b$
    - 比如， $\delta_{3}(s) = 0.1008，对应的是\delta_{2}(s)*a_{se}*b_e(o_3) = 0.1008，则\Psi_3(s) = e$
        
![维特比算法实例](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/维特比算法实例.jpg)


- 上述3个基本问题的联系
    - 首先，要学会用前向算法和后向算法算观测序列出现的概率，
    - 然后，用Baum-Welch算法求参数的时候，某些步骤是需要用到前向算法和后向算法的，
    - 最后，计算得到参数后，我们就可以用来做预测了。
    - 因此，三个基本问题，它们是渐进的，解决NLP问题，应用HMM模型做解码任务应该是最终的目的。

<!-- #region -->
#### Transformer
- 简介
    - Transformer本身是一个典型的encoder-decoder模型，Encoder端和Decoder端均有6个Block
    - Encoder端的Block包括两个模块，多头self-attention模块以及一个前馈神经网络模块；
    - Decoder端的Block包括三个模块，多头self-attention模块，多头Encoder-Decoder attention交互模块，以及一个前馈神经网络模块；
    - 需要注意：**Encoder端和Decoder端中的每个模块都有残差层和Layer Normalization层。即下图的Add & Norm**
    
    
- transformer模型由编码器（6层）和解码器（6层）组成，下文主要介绍编码器和解码器的内容；
    - 编码器，将自然语言序列映射成隐藏层表示（hidden state）；
    - 解码器，将隐藏层映射为自然语音序列，进而解决各类问题。
    - 原transformer模型为了加速residual connections，**将所有子层以及embedding层的输出都设置为512**
    - 2个堆叠式编码器和解码器组成的Transformer中，是需要**Encoder-Decoder Attention**的

![transformer_结构](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/transformer结构图.png)
<!-- #endregion -->

<!-- #region -->
#### BERT模型
1. Bert 的是怎样预训练的 
    - MLM：将完整句子中的部分字 mask，预测该 mask 词 
    - NSP：为每个训练前的例子选择句子 A 和 B 时，50% 的情况下 B 是真的在 A 后面的下一个句子， 50% 的情况下是来自语料库的随机句子，进行二分预测是否为真实下一句 在数据中随机选择 15% 的标记，其中 80%被换位[mask]，10%不变、10%随机替换其他单 词，原因是什么
        - mask 只会出现在构造句子中，当真实场景下是不会出现 mask 的，全 mask 不 match 句型了
        - 随机替换也帮助训练修正了[unused]和[UNK]
        - 强迫文本记忆上下文信息


2. 为什么 BERT 有 3 个嵌入层，它们都是如何实现的
    - input_id 是语义表达，和传统的 w2v 一样，方法也一样的 lookup
    - segment_id 是辅助 BERT 区别句子对中的两个句子的向量表示，从[1,embedding_size]里 面 lookup
    - position_id 是为了获取文本天生的有序信息，否则就和传统词袋模型一样了，从 [511,embedding_size]里面 lookup 
    
    
3. Bert里面为什么用layer normalization，而不用batch normalization，分别讲一下这两个啥意思。
    - Batch Normalization 是对这批样本的同一维度特征做归一化， 
    - Layer Normalization 是对这单个样本的所有维度特征做归一化。
    - 区别：
        - LN中同层神经元输入拥有相同的均值和方差，不同的输入样本有不同的均值和方差；
        - BN中则针对不同神经元输入计算均值和方差，同一个batch中的输入拥有相同的均值和方差。
    - LN不依赖于batch的大小和输入sequence的长度，因此可以用于batch size为1和RNN中sequence的normalize操作。
    
    
4. Bert里面为什么Q，K，V要用三个不同的矩阵，用一个不是也行吗。
    - 如果使用相同的矩阵，相同量级的情况下，q和k进行点积的值会是最大的，进行softmax的加权平均后，该词所占的比重会最大，使得其他词的比重很少，**无法有效利用上下文信息来增强当前词的语义表示，**
    - 而使用不同的QKV后，会很大程度减轻上述的影响。
    
    
5. Bert和transformer讲一下。
    - 1 **bert只有transformer的encode 结构 ，是生成语言模型**
    - 2 bert 加入了输入句子的 mask机制，在输入的时候会随机mask
    - 3 模型接收两个句子作为输入，并且预测其中第二个句子是否在原始文档中也是后续句子 可以做对话机制的应答。
    - 4 在训练 BERT 模型时，Masked LM 和 Next Sentence Prediction 是一起训练的，目标就是要最小化两种策略的组合损失函数。


6. attention为什么要除以根号下dk？
    - QK进行点击之后，值之间的方差会较大，也就是大小差距会较大；
    - 如果直接通过Softmax操作，会导致大的更大，小的更小；进行缩放，会使参数更平滑，训练效果更好。
<!-- #endregion -->

### 实现demo


#### HMM进行分词
- 关于HMM，详细内容可参考[EM&HMM和CRF模型](https://github.com/w666x/blog_items/blob/main/04_nlp/EM&HMM&CRF模型.md)
- 本质上看，分词可以看做是一个为文本中每个字符分类的过程，例如我们现在定义两个类别：
    - E代表词尾词，B代表非词尾词，
    - 于是分词“你/现在/应该/去/幼儿园/了”可以表达为：你E现B在E应B该E去E幼B儿B园E了B，
    - 分类完成后只需要对结果进行“解读”就可以得到分词结果了。


- 以下面例子做说明，当我们计算到第三个汉字“是”后停止，
    - 首先，应用上式算法步骤中的终止过程，计算 $P^*和i_T^*$的值，为 $P^*=0.1008、i_3^*=s$
    - 然后，进行回溯，即： $i_2^* = \Psi_3(s) = e \rightarrow i_1^* = \Psi_2(e) = b$
    - 综上，即得到前三个状态序列为：(b, e, s)
    
![维特比算法实例](https://cdn.jsdelivr.net/gh/w666x/image/NLP_base/维特比算法实例.jpg)
