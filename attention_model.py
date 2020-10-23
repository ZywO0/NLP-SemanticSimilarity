import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


class create_attention_model(nn.Module):
    def __init__(self, batch_size, max_query_len,  max_doc_len, vocab_size, embedding_matrix,
                 nb_filters, embed_size=300, dropout_rate=0.1, num_classes=2):
        super(create_attention_model, self).__init__()
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.embedding_matrix = embedding_matrix
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.nb_filters = nb_filters
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.num_class = num_classes
        # 词嵌入层
        self.embedding_layer = self.add_embed_layer(self.embedding_matrix, self.vocab_size['word'], self.embed_size).to(device)
        self.dropout_layer = nn.Dropout(self.dropout_rate).to(device)
        # CNN做编码层
        self.conv_layer = nn.Conv1d(in_channels=self.embed_size, kernel_size=2, stride=1,
                                    padding=0, out_channels=self.nb_filters)
        # 输出分类层
        self.dense_layer = nn.Linear(84, 2).cuda()
        nn.init.xavier_uniform(self.dense_layer.weight.data, gain=1)
        nn.init.constant(self.dense_layer.bias.data, 0.1)

    def add_embed_layer(self, vocab_emb, vocab_size, embed_size):
        if vocab_emb is not None:
            # 预训练词向量
            embed_layer = nn.Embedding(vocab_size, embed_size)
            pretrained_weight = np.array(vocab_emb)
            embed_layer.weight.data.copy_(torch.from_numpy(pretrained_weight))
        else:
            # 随机初始化
            print("Embedding with random weights")
            embed_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        return embed_layer

    def max_pooling(self, x):
        return torch.max(x, dim=1)[0]

    def forward(self, query_word_input, doc_word_input):
        # 经过词嵌入层，获取词向量
        query_embedding = self.embedding_layer(query_word_input)
        query_embedding = self.dropout_layer(query_embedding)
        doc_embedding = self.embedding_layer(doc_word_input)
        doc_embedding = self.dropout_layer(doc_embedding)
        # 经过卷积层
        query_embedding = query_embedding.permute(0, 2, 1)
        doc_embedding = doc_embedding.permute(0, 2, 1)
        query_conv_tensor = self.conv_layer(query_embedding)
        query_drop_tensor = F.relu(self.dropout_layer(query_conv_tensor))
        doc_conv_tensor = self.conv_layer(doc_embedding)
        doc_drop_tensor = F.relu(self.dropout_layer(doc_conv_tensor))
        # print("query:", query_drop_tensor.size())
        # print("doc:", doc_drop_tensor.size())
        # 拼接
        feature_list = [query_drop_tensor, doc_drop_tensor]
        concat_tensor = torch.cat(feature_list, dim=-1)
        # print("concat:", concat_tensor.size())
        max_sim = self.max_pooling(concat_tensor)
        # print("max_sim:", max_sim.size())       # [128, 84]
        # 经过全连接层
        prediction = self.dense_layer(max_sim)
        # print("prediction:", prediction.size())     # [128, 2]

        return prediction










