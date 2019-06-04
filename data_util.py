#coding=utf-8
import pandas as pd
from itertools import chain
import jieba
from tqdm import tqdm
import numpy as np
import random
import os
import string
import pickle
from collections import Iterator #******************************************************************************

class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        self.encoderSeqs = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.weights = []


class TextData:

    def __init__(self, args):

        self.args = args

        self.sr_word2id = None
        self.sr_id2word = None
        self.sr_line_id = None

        self.train_samples = []  # train samples with id

        self.padToken = "<pad>"  # Padding
        self.goToken = "<go>"  # Start of sequence
        self.eosToken = "<eos>"  # End of sequence
        self.unknownToken = "<unk>"  # Word dropped from vocabulary
        self.numToken = 4 # Num of above Tokens

        self.load_data()
        self.build_word_dict()
        self.generate_conversations()

        # generate batches
        self.batch_index = 0
        self.epoch_completed = 0


    def load_data(self):
        if not os.path.exists(self.args.sr_word_id_path) or not os.path.exists(self.args.train_samples_path):
            # 读取 douban_single_turn.tsv 文件
            print("开始读取数据")
            self.lines = pd.read_csv(self.args.dialog, sep=" ", names=["question", "answer"], dtype={"question":str,"answer":str,}, engine="python",encoding='utf-8')
        			
            print("数据读取完毕")

    def build_word_dict(self):
        if not os.path.exists(self.args.sr_word_id_path):
            # 得到word2id和id2word两个词典
            print("开始构建词典")			
            words = self.lines.question
            words = words.append(self.lines.answer,ignore_index=True)
            words = words.to_frame(name="QA")
            words["QA"] = words["QA"].apply(str)			
            words["QA"] = words["QA"].apply(lambda conv : self.chinese_list(conv))			

            words = words.QA.values
            words = list(chain(*words))
	        			
            sr_words_count = pd.Series(words).value_counts()
            # 筛选出 出现次数 大于 1 的词作为 vocabulary
            sr_words_size = np.where(sr_words_count.values > self.args.vacab_filter)[0].size
            sr_words_index = sr_words_count.index[0:sr_words_size]

            self.sr_word2id = pd.Series(range(self.numToken, self.numToken + sr_words_size), index=sr_words_index)
            self.sr_id2word = pd.Series(sr_words_index, index=range(self.numToken, self.numToken + sr_words_size))
            self.sr_word2id[self.padToken] = 0
            self.sr_word2id[self.goToken] = 1
            self.sr_word2id[self.eosToken] = 2
            self.sr_word2id[self.unknownToken] = 3
            self.sr_id2word[0] = self.padToken
            self.sr_id2word[1] = self.goToken
            self.sr_id2word[2] = self.eosToken
            self.sr_id2word[3] = self.unknownToken
            print("词典构建完毕")
            with open(os.path.join(self.args.sr_word_id_path), 'wb') as handle:
                data = {
                    'word2id': self.sr_word2id,
                    'id2word': self.sr_id2word,
                }
                pickle.dump(data, handle, -1)
        else:
            print("从{}载入词典".format(self.args.sr_word_id_path))
            with open(self.args.sr_word_id_path, 'rb') as handle:
                data = pickle.load(handle)
                self.sr_word2id = data['word2id']
                self.sr_id2word = data['id2word']

    def replace_word_with_id(self, conv):
        #conv = list(map(str.lower, conv))#对中文应该不需要，
        conv = list(map(self.get_word_id, conv.split(",")))
        # temp = list(map(self.get_id_word, conv))
        return conv


    def get_word_id(self, word):
        if word in self.sr_word2id:
            return self.sr_word2id[word]
        else:
            return self.sr_word2id[self.unknownToken]

    def get_id_word(self, id):
        if id in self.sr_id2word:
            return self.sr_id2word[id]
        else:
            return self.unknownToken

    def generate_conversations(self):

        if not os.path.exists(self.args.train_samples_path):
            # 将word替换为id
            # self.replace_word_with_id()
            print("开始生成训练样本")

            for i in range(len(self.lines.question)): 				
                question_id = self.replace_word_with_id(str(self.lines.question[i]))
                answer_id = self.replace_word_with_id(str(self.lines.answer[i]))
                valid = self.filter_conversations(question_id, answer_id)

                if valid :
                    temp = [question_id, answer_id]
                    self.train_samples.append(temp)

            print("生成训练样本结束")
            with open(self.args.train_samples_path, 'wb') as handle:
                data = {
                    'train_samples': self.train_samples
                }
                pickle.dump(data, handle, -1)
        else:
            with open(self.args.train_samples_path, 'rb') as handle:
                data = pickle.load(handle)
                self.train_samples = data['train_samples']
            print("从{}导入训练样本".format(self.args.train_samples_path))

    #创建停用词list
    def stopwordlist(filepath):##############################################################################################################
        stopwords = [line.strip() for line in open(filepath,'r',encoding='utf-8').readlines()]
        return stopwords			
    def word_tokenizer(self, sentence):##############################################################################################################
        # 中文分词
        words = nltk.word_tokenize(str(sentence))
        return words
		
    def chinese_list(self, sentence):
        # 将中文分词后的由词组组成的句子→[词组1，词组2，...，词组n]
        words = sentence.split(",")
        return words		

    def filter_conversations(self, question_id, answer_id):
        # 筛选样本， 首先将encoder_input 或 decoder_input大于max_length的conversation过滤
        # 其次将target中包含有UNK的conversation过滤
        valid = True
        valid &= len(question_id) <= self.args.maxLength
        valid &= len(answer_id) <= self.args.maxLength
        valid &= answer_id.count(self.sr_word2id[self.unknownToken]) == 0

        return valid

    def get_next_batches(self):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        self.shuffle()

        batches = []

        def gen_next_samples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, len(self.train_samples), self.args.batch_size):
                yield self.train_samples[i:min(i + self.args.batch_size, len(self.train_samples))]

        # TODO: Should replace that by generator (better: by tf.queue)

        for samples in gen_next_samples():
            batch = self.create_batch(samples)
            batches.append(batch)
        return batches

    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.train_samples)

    def create_batch(self, samples):
        batch = Batch()
        batch_size = len(samples)

        # Create the batch tensor
        for i in range(batch_size):
            # Unpack the sample
            sample = samples[i]
            batch.encoderSeqs.append(list(reversed(
                sample[0])))  # Reverse inputs (and not outputs), little trick as defined on the original seq2seq paper
            batch.decoderSeqs.append([self.sr_word2id[self.goToken]] + sample[1] + [self.sr_word2id[self.eosToken]])  # Add the <go> and <eos> tokens
            batch.targetSeqs.append(
                batch.decoderSeqs[-1][1:])  # Same as decoder, but shifted to the left (ignore the <go>)

            # Long sentences should have been filtered during the dataset creation
            assert len(batch.encoderSeqs[i]) <= self.args.maxLengthEnco
            assert len(batch.decoderSeqs[i]) <= self.args.maxLengthDeco

            # TODO: Should use tf batch function to automatically add padding and batch samples
            # Add padding & define weight
            batch.encoderSeqs[i] = [self.sr_word2id[self.padToken]] * (self.args.maxLengthEnco -
                            len(batch.encoderSeqs[i])) + batch.encoderSeqs[i]  # Left padding for the input

            batch.weights.append(
                [1.0] * len(batch.targetSeqs[i]) + [0.0] * (self.args.maxLengthDeco - len(batch.targetSeqs[i])))
            batch.decoderSeqs[i] = batch.decoderSeqs[i] + [self.sr_word2id[self.padToken]] * (
            self.args.maxLengthDeco - len(batch.decoderSeqs[i]))
            batch.targetSeqs[i] = batch.targetSeqs[i] + [self.sr_word2id[self.padToken]] * (
            self.args.maxLengthDeco - len(batch.targetSeqs[i]))

        # Simple hack to reshape the batch
        encoderSeqsT = []  # Corrected orientation
        for i in range(self.args.maxLengthEnco):
            encoderSeqT = []
            for j in range(batch_size):
                encoderSeqT.append(batch.encoderSeqs[j][i])
            encoderSeqsT.append(encoderSeqT)
        batch.encoderSeqs = encoderSeqsT

        decoderSeqsT = []
        targetSeqsT = []
        weightsT = []
        for i in range(self.args.maxLengthDeco):
            decoderSeqT = []
            targetSeqT = []
            weightT = []
            for j in range(batch_size):
                decoderSeqT.append(batch.decoderSeqs[j][i])
                targetSeqT.append(batch.targetSeqs[j][i])
                weightT.append(batch.weights[j][i])
            decoderSeqsT.append(decoderSeqT)
            targetSeqsT.append(targetSeqT)
            weightsT.append(weightT)
        batch.decoderSeqs = decoderSeqsT
        batch.targetSeqs = targetSeqsT
        batch.weights = weightsT

        return batch

    def sentence2enco(self, sentence):
        """Encode a sequence and return a batch as an input for the model
        Return:
            Batch: a batch object containing the sentence, or none if something went wrong
        """

        if sentence == '':
            return None

        # First step: Divide the sentence in token
        tokens = jieba.cut(sentence)		
        i = 0

        # Second step: Convert the token in word ids
        wordIds = []				
		
        for token in tokens:	
            if token in self.sr_word2id:			
                wordIds.append(self.sr_word2id[token])  # Create the vocabulary and the training sentences
            else:
                wordIds.append(self.sr_word2id[self.unknownToken])		
                # Third step: creating the batch (add padding, reverse)
            i += 1
			
        if i > self.args.maxLength:
            return None       			

        batch = self.create_batch([[wordIds, []]])  # Mono batch, no target output

        return batch

    def deco2sentence(self, decoderOutputs):
        """Decode the output of the decoder and return a human friendly sentence
        decoderOutputs (list<np.array>):
        """
        sequence = []

        # Choose the words with the highest prediction score
        for out in decoderOutputs:		
            sequence.append(np.argmax(out))  # Adding each predicted word ids

        return sequence

    def sequence2str(self, sequence, clean=False, reverse=False):
        """Convert a list of integer into a human readable string
        Args:
            sequence (list<int>): the sentence to print
            clean (Bool): if set, remove the <go>, <pad> and <eos> tokens
            reverse (Bool): for the input, option to restore the standard order
        Return:
            str: the sentence
        """

        if not sequence:
            return ''

        if not clean:
            return ' '.join([self.sr_id2word[idx] for idx in sequence])

        sentence = []
        for wordId in sequence:			
            if wordId == self.sr_word2id[self.eosToken]:  # End of generated sentence		
                break
            elif wordId != self.padToken and wordId != self.goToken:		
                sentence.append(self.sr_id2word[wordId])

        if reverse:  # Reverse means input so no <eos> (otherwise pb with previous early stop)
            sentence.reverse()

        return self.detokenize(sentence)

    def detokenize(self, tokens):
        """Slightly cleaner version of joining with spaces.
        Args:
            tokens (list<string>): the sentence to print
        Return:
            str: the sentence
        """	
        return ''.join([t for t in tokens])

if __name__ == '__main__':
    class args:
        def __init__(self):
            self.dialog = "../../data/douban_single_turn.tsv"
    args = args()
    textData = TextData(args)
