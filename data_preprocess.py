# -*- coding:UTF-8 -*-
import time
from string import punctuation
import codecs
import re
import jieba
import os
import numpy as np
import gensim
from collections import Counter


PAD_WORD_INDEX = 0
OOV_WORD_INDEX = 1
MAX_WORD_LENGTH = 100


def split_sent(sent, qrepr):
    """将sent切分成tokens"""
    if qrepr == 'word':
        return [token for token in jieba.cut(sent)]
    else:
        raise Exception("Unrecognized represention %s !" % qrepr)


def read_sentences(path, vocab, is_train, repr='word', test_vocab=None):
    """将输入转换为id，创建词表"""
    question = []
    max_len = 0
    punc = punctuation + u'1-9a-zA-Z.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'
    with codecs.open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            #  去标点后的句子
            line = re.sub(r"[{}]+".format(punc), " ", line)
            # 分词之后的句子
            q_tokens = split_sent(line, repr)
            token_ids = []
            # 根据最大长度进行截短
            if len(q_tokens) > max_len:
                max_len = len(q_tokens)
            # 创建词表
            for token in q_tokens:
                if token not in vocab[repr]:
                    if is_train:
                        vocab[repr][token] = len(vocab[repr])
                    elif repr == 'word' and token not in test_vocab[repr]:
                        test_vocab[repr][token] = len(vocab[repr]) + len(test_vocab[repr])
                # 上面是完成vocab和test_vocab词表，下面是将对应的id找出来填入token_ids
                if token in vocab[repr]:
                    token_ids.append(vocab[repr][token])
                elif repr == 'word':
                    token_ids.append(test_vocab[repr][token])
                else:
                    token_ids.append(OOV_WORD_INDEX)
            # 转为id后的句子
            # print("token_ids:", token_ids)
            question.append(token_ids)
    return question, max_len


def select_best_length(path, limit_rate=0.95):
    """选择最佳的样本max_length"""
    len_list = []
    max_length = 0
    cover_rate = 0.0
    with codecs.open(path, "r", encoding='utf-8') as f:
        for line in f:
            len_list.append(len(line))
        all_sent = len(len_list)
        sum_length = 0
        len_dict = Counter(len_list).most_common()
        ## len_dict :[(len(q1):count(q1)))]
        for i in len_dict:
            sum_length += i[0] * i[1]
        average_length = sum_length/all_sent
        for i in len_dict:
            rate = i[1]/all_sent
            cover_rate += rate
            if cover_rate >= limit_rate:
                max_length = i[0]
                break
    print("max_length: ", max_length)
    return max_length


def read_relevance(path):
    """ 加载label文件"""
    sims = []
    if os.path.exists(path):
        with open(path) as f:
            for i, line in enumerate(f):
                sims.append(int(line.strip()))
    print("sims:", sims[0:5])
    return sims


def pad_sentences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0):
    """按对长度短的句子最大长度补全句子"""
    if not hasattr(sequences, '__len__'):
        raise ValueError('Sequences must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('sequences must be a list of iterables.'
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))
    num_samples = len(sequences)
    if maxlen is None:      # 但是传进来的maxlen不是None
        maxlen = np.max(lengths)

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s"'
                             'not understood' % truncating)

        # check 'trunc' has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s'
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def invert_dict(dict):
    dict_inv = [""] * (max(dict.values()) + 1)
    for word, index in dict.items():
        dict_inv[index] = word
    return dict_inv

def gen_data(path, datasets, vocab, test_vocab, is_train, max_query_len, max_doc_len):
    if is_train:
        vocab['word']['PAD_WORD_INDEX'] = PAD_WORD_INDEX
        vocab['word']['OOV_WORD_INDEX'] = OOV_WORD_INDEX

    query_word_list, doc_word_list= [], []
    all_sim_list = []

    for data_name in datasets:
        print(datasets)
        data_folder = "%s/%s" % (path, data_name)
        print(data_folder, data_name)
        print("creating datasets %s" % data_name)
        t = time.time()
        # 转为id输入
        q1_word_list, max_q1_word_len = read_sentences('%s/a.toks' % data_folder, vocab, is_train, 'word', test_vocab=test_vocab)
        q2_word_list, max_q2_word_len = read_sentences('%s/b.toks' % data_folder, vocab, is_train, 'word', test_vocab=test_vocab)
        if is_train:
            max_query_len['word'] = max(max_query_len['word'], min(max_q1_word_len, MAX_WORD_LENGTH))
            max_doc_len['word'] = max(max_doc_len['word'], min(max_q2_word_len, MAX_WORD_LENGTH))
        query_word_list.extend(q1_word_list)
        doc_word_list.extend(q2_word_list)
        sim_list = read_relevance("%s/sim.txt" % data_folder)
        all_sim_list.extend(sim_list)
        print("q1_max_word_len: %d, q2_max_word_len: %d, len limit: (%d, %d)" %
              (max_q1_word_len, max_q2_word_len, max_query_len['word'], max_doc_len['word']))
        print('creating dataset done : %d' % (time.time() - t))

    data = {'sim': np.array(all_sim_list)}
    data['query_word_input'] = pad_sentences(query_word_list, maxlen=max_query_len['word'],
                                             value=PAD_WORD_INDEX, padding='post', truncating='post')

    data['doc_word_input'] = pad_sentences(doc_word_list, maxlen=max_doc_len['word'],
                                           value=PAD_WORD_INDEX, padding='post', truncating='post')

    return data


def get_word_vector(entity_model, word):
    if type(entity_model) == tuple:
        # 如果是个三元组，找出word对应在vocab中的id,再根据Id找出emb
        vocab, emb = entity_model
        wid = vocab[word]
        return emb[wid]
    else:
        if word in entity_model:
            return entity_model[word]
        else:
            # 不在预训练词向量中，属于OOV
            return None


def construct_vocab_emb(data_path, train_vocab, test_vocab, embed_size, qrepr, base_embed_path, type='wiki'):
    train_vocab_emb, test_vocab_emb = None, None
    # 创建文件夹放OOV词语(out-of-vocabulary)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    f = open("%s/OOV_words.txt" % data_path, "w")
    # 加载预训练词向量
    print("Load %s word embedding..." % type)
    if type == 'wiki':
        assert base_embed_path.endswith("sgns.wiki.bigram-char")
        entity_model = gensim.models.KeyedVectors.load_word2vec_format(base_embed_path, binary=False, unicode_errors='ignore')
    print("Building embedding matrix from base embedding at %s ..." % base_embed_path)
    cnt_oov = 0
    # 创建词向量矩阵enbedding matrix
    train_vocab_emb = np.zeros((len(train_vocab), embed_size))
    test_vocab_emb = np.zeros((len(test_vocab), embed_size))
    print("train vocab size: %d, test vocab size: %d" %(len(train_vocab), len(test_vocab)))
    for word in train_vocab:
        wid = train_vocab[word]
        if wid != PAD_WORD_INDEX:
            emb = get_word_vector(entity_model, word)
            if emb is None:
                # 对OOV的处理，随机初始化
                cnt_oov += 1
                emb = np.random.rand(embed_size).astype(np.float32)
                emb = emb * 0.1
                f.write(word + '\n')
            train_vocab_emb[wid] = emb
            # print("train_emb:", emb)
    for word in test_vocab:
        # print("test_wid:", test_vocab[word])
        wid = test_vocab[word] - len(train_vocab)
        emb = get_word_vector(entity_model, word)
        if emb is None:
            cnt_oov += 1
            emb = np.random.rand(embed_size).astype(np.float32)
            emb = emb * 0.1
            f.write(word + '\n')
        test_vocab_emb[wid] = emb

    print("OOV words: %d" % cnt_oov)
    f.close()
    # print("PAD:", train_vocab_emb[PAD_WORD_INDEX])      # 是全0
    return train_vocab_emb, test_vocab_emb
















