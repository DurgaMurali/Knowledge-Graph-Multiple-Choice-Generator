import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import Attention_layer
import numpy as np
import knowledge_graph as kg
import random

random.seed(256)

class Answer_filtering_module(torch.nn.Module):
    def __init__(self, entity_embeddings, embedding_dim, vocab_size, word_dim, hidden_dim, fc_hidden_dim, relation_dim,
                 head_bn_filepath, score_bn_filepath):
        super(Answer_filtering_module, self).__init__()
        self.relation_dim = relation_dim
        self.loss_criterion = torch.nn.BCELoss(reduction='sum')
        # hidden_dim * 2 is the BiLSTM + attention layer output
        self.fc_lstm2hidden = torch.nn.Linear(hidden_dim * 2, fc_hidden_dim, bias=True)
        torch.nn.init.xavier_normal_(self.fc_lstm2hidden.weight.data)
        torch.nn.init.constant_(self.fc_lstm2hidden.bias.data, val=0.0)
        self.fc_hidden2relation = torch.nn.Linear(fc_hidden_dim, self.relation_dim, bias=False)
        torch.nn.init.xavier_normal_(self.fc_hidden2relation.weight.data)
        self.entity_embedding_layer = torch.nn.Embedding.from_pretrained(torch.tensor(entity_embeddings), freeze=True)
        self.word_embedding_layer = torch.nn.Embedding(vocab_size, word_dim)
        self.BiLSTM = torch.nn.LSTM(word_dim, hidden_dim, 1, bidirectional=True, batch_first=True)
        self.softmax_layer = torch.nn.LogSoftmax(dim=-1)
        self.attention_layer = Attention_layer(hidden_dim=2 * hidden_dim, attention_dim=4 * hidden_dim)
        self.head_bn = torch.nn.BatchNorm1d(2)
        head_bn_params_dict = np.load(head_bn_filepath, allow_pickle=True)
        head_bn_params_dict = head_bn_params_dict.item()
        self.head_bn.weight = torch.nn.Parameter(torch.from_numpy(head_bn_params_dict['weight']).float())
        self.head_bn.bias = torch.nn.Parameter(torch.from_numpy(head_bn_params_dict['bias']).float())
        self.score_bn = torch.nn.BatchNorm1d(2)
        score_bn_params_dict = np.load(score_bn_filepath, allow_pickle=True)
        score_bn_params_dict = score_bn_params_dict.item()
        self.score_bn.weight = torch.nn.Parameter(torch.from_numpy(score_bn_params_dict['weight']).float())
        self.score_bn.bias = torch.nn.Parameter(torch.from_numpy(score_bn_params_dict['bias']).float())

    def complex_scorer(self, head, relation):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        head = self.head_bn(head)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]
        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.entity_embedding_layer.weight, 2, dim=1)
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = torch.stack([re_score, im_score], dim=1)
        score = self.score_bn(score)
        score = score.permute(1, 0, 2)
        re_score = score[0]
        im_score = score[1]
        return torch.sigmoid(torch.mm(re_score, re_tail.transpose(1, 0)) + torch.mm(im_score, im_tail.transpose(1, 0)))

    def forward(self, question, questions_length, head_entity, tail_entity, max_sent_len, loader):
        embedded_question = self.word_embedding_layer(question)
        packed_input = pack_padded_sequence(embedded_question, questions_length, batch_first=True)
        packed_output, _ = self.BiLSTM(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, padding_value=0.0, total_length=max_sent_len)
        output = self.attention_layer(output.permute(1, 0, 2), questions_length)
        output = torch.nn.functional.relu(self.fc_lstm2hidden(output))
        relation_embedding = self.fc_hidden2relation(output)
        pred_answers_score = self.complex_scorer(self.entity_embedding_layer(head_entity), relation_embedding)

        bce_loss = self.loss_criterion(pred_answers_score, tail_entity)

        # Find entity type of correct answer 
        answers_entity = [torch.argmax(tail) for tail in tail_entity]
        answers_entity_type = [kg.ENTITY_2_ENTITY_TYPE[ans.item()] for ans in answers_entity]

        num_samples = 20
        positive_samples = []
        negative_samples = []
        for idx in range(len(answers_entity)):
            ans_entity_type = answers_entity_type[idx]
            similar_entities = kg.ENTITY_TYPE_2_ENTITY[ans_entity_type].copy()
            # Remove correct answer
            similar_entities.remove(answers_entity[idx].item())
            if len(similar_entities) > num_samples:
                rand_idxs = random.sample(range(0, len(similar_entities)), num_samples)
            else:
                rand_idxs = [idx for idx in range(len(similar_entities))]

            positive_embeddings = [self.entity_embedding_layer(torch.tensor(similar_entities[idx], dtype=torch.long)) for idx in rand_idxs]
            positive_samples.append(positive_embeddings)

            for type in kg.ENTITY_TYPE_2_ENTITY.keys():
                # If type not same as answer entity and not equal to rating, votes and tags
                if type != ans_entity_type:
                    different_entities = kg.ENTITY_TYPE_2_ENTITY[type].copy()
                    if len(different_entities) > num_samples:
                        rand_idxs = random.sample(range(0, len(different_entities)), num_samples)
                    else:
                        rand_idxs = [idx for idx in range(len(different_entities))]

                    negative_embeddings = [self.entity_embedding_layer(torch.tensor(different_entities[idx], dtype=torch.long)) for idx in rand_idxs]
                    negative_samples.append(negative_embeddings)

        ans_entity_embedding = [self.entity_embedding_layer(ans) for ans in answers_entity]

        margin = 0.0001
        contrastive_loss = 0.0       
        for idx in range(len(ans_entity_embedding)):
            for pos in positive_samples[idx]:
                pos_dist = torch.nn.functional.pairwise_distance(ans_entity_embedding[idx].unsqueeze(0), pos.unsqueeze(0))
                for neg in negative_samples[idx]:
                    neg_dist = torch.nn.functional.pairwise_distance(ans_entity_embedding[idx].unsqueeze(0), neg.unsqueeze(0))
                    contrastive_loss += torch.clamp(pos_dist+margin-neg_dist, min=0)
        
        triple_loss_factor = 0.0005
        loss = bce_loss + triple_loss_factor*contrastive_loss

        return loss
    

    def get_ranked_top_k(self, question, questions_length, head_entity, max_sent_len, K=5):
        embedded_question = self.word_embedding_layer(question)
        packed_input = pack_padded_sequence(embedded_question, questions_length, batch_first=True)
        packed_output, _ = self.BiLSTM(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, padding_value=0.0, total_length=max_sent_len)
        output = self.attention_layer(output.permute(1, 0, 2), questions_length)
        output = torch.nn.functional.relu(self.fc_lstm2hidden(output))
        relation_embedding = self.fc_hidden2relation(output)
        pred_answers_score = self.complex_scorer(self.entity_embedding_layer(head_entity), relation_embedding)
        return torch.topk(pred_answers_score, k=K, dim=-1, largest=True, sorted=True)  