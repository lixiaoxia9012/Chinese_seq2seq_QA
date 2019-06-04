class args:
    def __init__(self):
        # data args
        self.dialog = "data/douban_single_turn.txt"
        self.train_samples_path = "data_utils/train_samples.pkl"
        self.sr_word_id_path = "data_utils/word_id.pkl"
        self.vacab_filter = 1   # 筛选出 出现次数 大于 1 的_单字/词组_作为 词典中的 单字/词组；
        self.corpus_name = "seq2seq_model"

        # model args
        self.maxLengthEnco = 15 #最大的_编码_的句子中词组的数量；
        self.maxLengthDeco = 17 #最大的_解码_的句子中词组的数量 +2 ；
        self.maxLength = 15   #测试/交互模式：问题句中的词组的最大数量；生成训练样本时要求问句/答句中的单字/词组的个数要小于这里设置的数量
        self.embedding_size = 64
        self.hidden_size = 512
        self.rnn_layers = 2
        self.dropout = 0.9
        self.batch_size = 256
        self.learning_rate = 0.002
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-08
        self.softmaxSamples = 0

        # train args
        self.log_device_placement = False
        self.allow_soft_placement = True #将无法在GPU上的操作放回CPU；；
        self.num_checkpoints = 100
        self.epoch_nums = 30   #训练时会根据语料自动生成epoch(比如每个epoch包含小单位的个数是50多个/900多个)，这里设定的是训练中epoch运行的次数；；发现每次运行的时间并不相等_可能与语料的难易/多少/进行第几次训练有关；；；
        self.checkpoint_every = 100
        self.evaluate_every = 100
        self.test = None
        #self.test = 'interactive'
        # self.test = 'web_interface'