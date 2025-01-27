import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, ) - 0.5
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)

        if prob.dim() > 2:
            # used for 3d-conv:  N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            N, C = logit.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]

        ori_shp = target.shape
        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.squeeze(-1))
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)

        return loss




class Attention(nn.Module):       #x:[batch, seq_len, hidden_dim*2]
    """
        此注意力的计算步骤：
        1.将输入（包含lstm的所有时刻的状态输出）和w矩阵进行矩阵相乘，然后用tanh压缩到(-1, 1)之间
        2.然后再和矩阵u进行矩阵相乘后，矩阵变为1维，然后进行softmax变化即得到注意力得分。
        3.将输入和此注意力得分线性加权，即相当于将所有时刻的状态进行了一个聚合。
    """
    def __init__(self, hidden_size, need_aggregation=True):
        super().__init__()
        self.need_aggregation = need_aggregation
        # 不双向的话就不用乘2
        self.w = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        nn.init.uniform_(self.w, -0.1, 0.1)
        nn.init.uniform_(self.u, -0.1, 0.1)

    def forward(self, x):
        device = x.device
        self.w = self.w.to(device)
        self.u = self.u.to(device)

        u = torch.tanh(torch.matmul(x, self.w))         #[batch, seq_len, hidden_size*2]
        score = torch.matmul(u, self.u)                   #[batch, seq_len, 1]
        att = F.softmax(score, dim=1)
        # 下面操作即线性加权
        scored_x = x * att                              #[batch, seq_len, hidden_size*2]

        # 因为词encoder和句encoder后均带有attention机制，而我需要做的是代码行级缺陷检测，
        # 所以句encoder后我不做聚合，相当于将每个代码行看做一个样本来传入全连接分类。
        if self.need_aggregation:
            context = torch.sum(scored_x, dim=1)                  #[batch, hidden_size*2]
            return context
        else:
            return scored_x


class AttentionFusion(nn.Module):

    def __init__(self):
        super(AttentionFusion, self).__init__()

    def forward(self, Q, K):
 
        batch_size, sentence_nums, d_k = K.size()

     
        Q_expanded = Q.unsqueeze(1)  # (batch_size, 1, d_k)

     
        scores = torch.matmul(Q_expanded, K.transpose(-2, -1)) / math.sqrt(d_k)

    
        attn_weights = torch.softmax(scores, dim=-1)

     
        output = torch.matmul(attn_weights, K)  # (batch_size, 1, d_k)

   
        output = output.squeeze(1)  # (batch_size, d_k)
        output = output.unsqueeze(1).expand(-1, sentence_nums, -1)  # (batch_size, sentence_nums, d_k)

     
        fused_output = K + output  # (batch_size, sentence_nums, d_k)

        return fused_output


class TokenLevelNetwork(nn.Module):
    def __init__(self, embed_size, hidden_size, num_heads, sentence_length, sentence_nums, num_classes=2):
        super(TokenLevelNetwork, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.sentence_length = sentence_length
        self.sentence_nums = sentence_nums
        self.num_classes = num_classes

        self.projection = nn.Linear(embed_size, hidden_size)

        self.conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.self_attention_sentence = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        self.sentence_interaction = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size // 2, num_classes)
        # )
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size // 2),
        #     nn.Tanh(),
        #     nn.Linear(hidden_size // 2, num_classes),
        #     nn.Dropout(0.5)
        # )


    def forward(self, x, mask=None):
        batch_size, sentence_nums, sentence_length, embed_size = x.size()

        x = x.view(-1, sentence_length, embed_size)
        x = self.projection(x)
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = F.relu(x)
        x = x.transpose(1, 2)

        if mask is not None:
            mask = mask.view(-1, sentence_length)
        x, _ = self.self_attention_sentence(x, x, x, key_padding_mask=mask)
        # x, _ = self.self_attention_sentence(x, x, x)

        # x = torch.mean(x, dim=1)
        x = x[:,0,:]


        if mask is not None:
            # Create a padding mask for the sentence interaction (this is optional depending on your data)
            sentence_padding_mask = mask.view(batch_size, sentence_nums, sentence_length).any(dim=2)
            sentence_padding_mask = sentence_padding_mask.view(-1, sentence_nums)  # Flatten it for interaction
        else:
            sentence_padding_mask = None
        x = x.view(batch_size, sentence_nums, -1)
        # x, _ = self.sentence_interaction(x, x, x)
        logits, _ = self.sentence_interaction(x, x, x, key_padding_mask=sentence_padding_mask)


 
        # x 1， 256， 768   manual_feature->Q x->K,v

        # logits = self.classifier(x)

        return logits

class HAN_MODEL(nn.Module):
    def __init__(self, embedding_layer):
        super().__init__()
        self.dropout = nn.Dropout(0.5)

        self.hidden_size = 256
        self.num_layers = 1
        self.bidirectional = True

        self.embedding = embedding_layer

        self.lstm1 = nn.LSTM(input_size=768,
                            hidden_size=512,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional,
                            batch_first=True)
        self.att1 = TokenLevelNetwork(embed_size=768, hidden_size=768,
                                         num_heads=8,sentence_length=64,
                                         sentence_nums=768,num_classes=2)

        self.fc1 = nn.Linear(1024, 256)
        self.relu = nn.Tanh()
        self.fc2 = nn.Linear(256, 2)
        # self.classifier = TokenLevelNetwork
        hidden_size = 1024
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 2),
            nn.Dropout(0.5)
        )

    def forward(self, x, manual_feature):
        # 根据x全零制造一个mask
        padding_mask = ~(x == 1)

        # input x : (bs, nus_sentences, nums_words)
        device = x.device
        x = self.embedding(x) # out x : (bs, nus_sentences, nums_words, embedding_dim)
        x = self.dropout(x)
        batch_size, num_sentences, num_words, emb_dim = x.shape
        hidden_size = 512
        num_directions = 2


        # batch sentence_length, 768
        x = self.att1(x, None)
        x, _ = self.lstm1(x)
        x = self.classifier(x)

        return x

class AttentionFusionNetwork(nn.Module):
    def __init__(self, embed_size, expert_size, output_size):
        super(AttentionFusionNetwork, self).__init__()
        # Linear layers to transform input dimensions
        self.query_layer = nn.Linear(expert_size, embed_size)  # To match dimensions
        self.key_layer = nn.Linear(embed_size, embed_size)
        self.value_layer = nn.Linear(embed_size, embed_size)

        # Output layer
        self.output_layer = nn.Linear(embed_size + expert_size, output_size)

    def forward(self, msg_tensor, expert_vector):
        # msg_tensor: (batch, 512, embed_size)
        # expert_vector: (batch, expert_size)

        # Prepare expert_vector to be the query
        query = self.query_layer(expert_vector).unsqueeze(1)  # (batch, 1, embed_size)

        # Compute keys and values from msg_tensor
        keys = self.key_layer(msg_tensor)  # (batch, 512, embed_size)
        values = self.value_layer(msg_tensor)  # (batch, 512, embed_size)

        # Compute attention scores
        attn_scores = torch.matmul(keys, query.transpose(-1, -2))  # (batch, 512, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, 512, 1)

        # Compute attention-weighted sum of values
        attn_output = torch.sum(attn_weights * values, dim=1)  # (batch, embed_size)

        # Concatenate attention output with expert vector
        fusion = torch.cat([attn_output, expert_vector], dim=-1)  # (batch, embed_size + expert_size)

        # Final output layer
        output = self.output_layer(fusion)  # (batch, output_size)
        return output

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.manual_dense = nn.Linear(config.feature_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj_new = nn.Linear(config.hidden_size + config.hidden_size, 2)

        self.attn_fusion = AttentionFusionNetwork(768, 768, 1536)

    def forward(self, features, manual_features=None, **kwargs):
        y = manual_features.float()  # [bs, feature_size]
        y = self.manual_dense(y)
        y = torch.tanh(y)

        # x = features[:, 0, :] # take <s> token (equiv. to [CLS])  [bs,hidden_size]
        # x = torch.cat((x, y), dim=-1)
        x = self.attn_fusion(features, y)

        x = self.dropout(x)
        x = self.out_proj_new(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder #  codebert (RobertaModel)
        self.config = config #
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)  # second select mlp
        self.args = args


        # ----------------------HAN-------------------------------
        self.han_word_embedding_layer = self.encoder.embeddings.word_embeddings
        self.han_locator = HAN_MODEL(embedding_layer=self.han_word_embedding_layer)

        # --------------------------------------------------------


        # self.fusion_fc = nn.Linear(4,2)




    def forward(self, inputs_ids, attn_masks, manual_features,
                labels, line_ids, line_label, output_attentions=None):
        # inputs_ids(batch, 512) msg + add code + del code
        # attn_masks(batch, 512)
        # manual_features(batch, 14)
        # labels(batch, 1)  0/1 判断这个样本是否为1
        # line_ids(batch, 256, 64) ; 256 code change line padding 256 ;64 each code line padding 64
        # line_label(batch, 256) 0/1 判断每一个line是不是有问题
        outputs = self.encoder(input_ids=inputs_ids, attention_mask=attn_masks, output_attentions=output_attentions)

        last_layer_attn_weights = outputs.attentions[self.config.num_hidden_layers - 1][:, :, 0].detach() if output_attentions else None
 
        logits = self.classifier(outputs[0], manual_features)  # outputs[0] 1 512 768

        han_logits = self.han_locator(line_ids, manual_features)

        # logits = self.fusion_fc(torch.cat((logits, han_outputs), dim=-1))
        logits = (logits+han_logits.mean(dim=1))/2


        if labels is not None:


            loss_dp = MultiFocalLoss(alpha=0.25, gamma=2, reduction='mean', num_class=2)
            loss1 = loss_dp(logits, labels)


            loss_dl = MultiFocalLoss(alpha=0.25, gamma=2, reduction='mean', num_class=2)
            loss2 = loss_dl(han_logits.reshape((-1, 2)), line_label.reshape((-1,)))

            # loss = (loss1 + loss2) / 2
            loss = loss1*self.args.dp_loss_weight + loss2*self.args.dl_loss_weight


            return loss, torch.softmax(logits,dim=1)[:, 1].unsqueeze(1), last_layer_attn_weights, torch.softmax(han_logits, dim=-1)[:, :, 1] # shape: (bs, line_nums:256)
        else:
            # return torch.sigmoid(logits)[:, 1].unsqueeze(1)
            return torch.softmax(logits, dim=1)[:, 1].unsqueeze(1)

