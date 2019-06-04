# Chinese_seq2seq_QA

测试/预测：
1. 确认： args.py 下面的参数有关训练还是测试/预测的选择正确，即为：
        #self.test = None
        self.test = 'interactive'
2. 然后：直接运行 training.py，就会弹出 单轮一问一答 对话框；
   模型采用的是（step为xxx，loss为xxx，perplexity为xxx），测试结果如下(过几天添加)：
    
        
训练：
1. 将本作品之前生成的模型删除或另存；
2. 修改 args.py 下面的相关参数，其中必须选择
     self.test = None
     #self.test = 'interactive'
3. 直接运行 training.py 即可。


项目有待完善的地方：
1. 修改训练时的关键参数；
2. 修改训练用到的 seq2seq 模型(tensorflow 中开源的 seq2seq 有 5 种，这里只尝试了1种)；
3. 修改对应的损失函数的选择（tensorflow 中用于 seq2seq 的损失模块有 2 种，这里只试了1 种）；
4. 框架上：
   测试/预测时，应该有 停用词和输出结果的 合成，待添加这部分；
   这里的中文语料已经分词过了，如需替换语料，需要加入分词部分，并根据数据结构修改相应的中文数据处理部分；
5. 使用更为丰富、质量更好的语料；

参考：
1. 大部分代码都是参考 https://github.com/Irvinglove/Seq2seq-QA(原始语料：英文)；
   本项目修改了中文原始语料的适配部分。
2. 中文语料来自： https://github.com/codemayq/chinese_chatbot_corpus；
3. 中文语料的处理，主要参考了：《基于 LSTM 的 Chatbot 实例》；
   网址： https://blog.csdn.net/zhangchen2449/article/details/80479202；
