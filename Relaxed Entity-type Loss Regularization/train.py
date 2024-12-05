import time
from dataloader import MetaQADataLoader, DEV_MetaQADataLoader
from model import Answer_filtering_module
import torch
import logging
from tqdm import tqdm
import os
from collections import OrderedDict
import numpy as np
import knowledge_graph

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# =========
KG_NAME = 'MetaQA'
# =========
HOPS = 1
assert HOPS in [1, 2, 3]
# =========
KG_HALF = False
# =========
if KG_NAME == 'MetaQA':
    QA_PAIRS_PATH = f'../QA_single_answer/MetaQA/qa_%s_{str(HOPS)}hop{"_half" if KG_HALF else ""}.txt'
else:
    QA_PAIRS_PATH = f'../QA/WebQuestionsSP/qa_%s_webqsp.txt'

qa_traindataset_path = QA_PAIRS_PATH % 'train'
qa_devdataset_path = QA_PAIRS_PATH % 'dev'
qa_testdataset_path = QA_PAIRS_PATH % 'test'

# ====dataloader prepare=====
BEST_OR_FINAL = 'best'
assert BEST_OR_FINAL in ['best', 'final']

KG_EMBED_PATH = f'../knowledge_graph_embedding_module/kg_embeddings/{KG_NAME}/' \
                f'{BEST_OR_FINAL}_checkpoint/%s'

score_bn_path = KG_EMBED_PATH % 'score_bn.npy'
head_bn_path = KG_EMBED_PATH % 'head_bn.npy'
R_path = KG_EMBED_PATH % 'R.npy'
E_path = KG_EMBED_PATH % 'E.npy'
entity_dict_path = KG_EMBED_PATH % 'entities_idx.dict'
relation_dict_path = KG_EMBED_PATH % 'relations_idx.dict'
batch_size = 128

qa_dataloader = MetaQADataLoader(entity_embed_path=E_path, entity_dict_path=entity_dict_path, relation_embed_path=R_path
                                 , relation_dict_path=relation_dict_path, qa_dataset_path=qa_traindataset_path,
                                 batch_size=batch_size)
word_idx = qa_dataloader.dataset.word_idx
qa_dev_dataloader = DEV_MetaQADataLoader(word_idx=word_idx, entity_dict_path=entity_dict_path,
                                         relation_dict_path=relation_dict_path,
                                         qa_dataset_path=qa_devdataset_path)
qa_test_dataloader = DEV_MetaQADataLoader(word_idx=word_idx, entity_dict_path=entity_dict_path,
                                          relation_dict_path=relation_dict_path,
                                          qa_dataset_path=qa_testdataset_path)
# ====model hyper-parameters prepare=====
entity_embeddings = qa_dataloader.dataset.entity_embeddings
print(type(entity_embeddings))
print(entity_embeddings.shape)
embedding_dim = entity_embeddings.shape[-1]
relation_embeddings = qa_dataloader.dataset.relation_embeddings
print(type(relation_embeddings))
print(relation_embeddings.shape)
relation_dim = relation_embeddings.shape[-1]
vocab_size = len(qa_dataloader.dataset.word_idx)
word_dim = 256
hidden_dim = 200
fc_hidden_dim = 400
load_from = ''
# ====model init=====
model = Answer_filtering_module(entity_embeddings=entity_embeddings, embedding_dim=embedding_dim, vocab_size=vocab_size,
                                word_dim=word_dim, hidden_dim=hidden_dim, fc_hidden_dim=fc_hidden_dim,
                                relation_dim=relation_dim,
                                head_bn_filepath=head_bn_path, score_bn_filepath=score_bn_path)
if load_from:
    model = torch.load(load_from)
model.to(device)
model.train()

# ====training hyper-parameters prepare=====
TRAINING_RESULTS_DIR = os.path.join('.', '_'.join([KG_NAME, "half" if KG_HALF else "full", str(HOPS), 'hop',
                                                   time.asctime().replace(' ', '_').replace(':', '_')]))
if not os.path.isdir(TRAINING_RESULTS_DIR):
    os.makedirs(TRAINING_RESULTS_DIR)
if not os.path.exists(os.path.join(TRAINING_RESULTS_DIR, 'word_idx.txt')):
    with open(os.path.join(TRAINING_RESULTS_DIR, 'word_idx.txt'), 'wt', encoding='utf-8') as outp:
        for word, idx in word_idx.items():
            outp.write(word + '\t' + str(idx) + '\n')
best_model_path = os.path.join(TRAINING_RESULTS_DIR, 'best_afm_model.pt')
final_model_path = os.path.join(TRAINING_RESULTS_DIR, 'final_afm_model.pt')
N_EPOCHS = 200
PATIENCE = 5
LR = 0.0001
adam_optimizer = torch.optim.Adam(model.parameters(), lr=LR)
LR_DECAY = 0.95
TEST_INTERVAL = 5
TEST_TOP_K = 5
NO_UPDATE = 0
best_val_score = 0
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=adam_optimizer, gamma=LR_DECAY)
logger = logging.getLogger()
logger.setLevel(logging.NOTSET)
file_logger = logging.FileHandler(os.path.join(TRAINING_RESULTS_DIR, 'training.log'), encoding='UTF-8')
file_logger.setLevel(logging.INFO)
file_logger.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(file_logger)
# ====training step=====
epoch_process = tqdm(qa_dataloader, total=len(qa_dataloader), unit=' batches')
epoch_dev_process = tqdm(qa_dev_dataloader, total=len(qa_dev_dataloader), unit=' batches')
epoch_test_process = tqdm(qa_test_dataloader, total=len(qa_test_dataloader), unit=' batches')


def correct_rate(head_entity, topK_entity_idx, answers, question, loader):
    """
    :param head_entity: number index
    :param topK_entity_idx: topK list[number]
    :param answers: list[number]
    :return:
    """
    points = 0
    for candid in topK_entity_idx:
        if candid != head_entity and candid in answers:
            points += 1
    return points / len(topK_entity_idx)


# ====training step=====
for epoch_idx in range(N_EPOCHS):
    epoch_idx += 1
    avg_epoch_loss = 0
    epoch_process.set_description('{}/{}'.format(epoch_idx, N_EPOCHS))
    for batch_questions_index, batch_questions_length, batch_head_entity, batch_answers, max_sent_len in epoch_process:
        model.zero_grad()
        loss = model(question=batch_questions_index.to(device), questions_length=batch_questions_length.to(device),
                     head_entity=batch_head_entity.to(device), tail_entity=batch_answers.to(device), max_sent_len=max_sent_len, loader=qa_dataloader)
        loss.backward()
        adam_optimizer.step()
        avg_epoch_loss += loss.item()
        epoch_process.set_postfix(
            OrderedDict(Epoch=epoch_idx, Batch_Loss=loss.item(), Sample_Loss=loss.item() / batch_size))
        epoch_process.update()
    logger.info(f'{epoch_idx}-th epoch: average_loss: {avg_epoch_loss / (len(qa_dataloader))}')
    if epoch_idx % TEST_INTERVAL == 0:
        model.eval()
        eval_correct_rate = 0
        for questions, batch_questions_index, batch_questions_length, batch_head_entity, batch_answers, max_sent_len in epoch_dev_process:
            ranked_topK_entity_idxs = model.get_ranked_top_k(batch_questions_index, batch_questions_length,
                                                             batch_head_entity, max_sent_len, K=TEST_TOP_K)
            ranked_topK_entity_idxs = ranked_topK_entity_idxs.indices.tolist()
            batch_head_entity = batch_head_entity.tolist()
            batch_correct_rate = np.sum(
                np.array([correct_rate(head_entity=head_entity, topK_entity_idx=ranked_topK_entity_idxs[idx],
                                       answers=batch_answers[idx], question=questions[idx], loader=qa_dev_dataloader) for idx, head_entity in
                          enumerate(batch_head_entity)]))
            eval_correct_rate += batch_correct_rate
        for questions, batch_questions_index, batch_questions_length, batch_head_entity, batch_answers, max_sent_len in epoch_test_process:
            ranked_topK_entity_idxs = model.get_ranked_top_k(batch_questions_index, batch_questions_length,
                                                             batch_head_entity, max_sent_len, K=TEST_TOP_K)
            ranked_topK_entity_idxs = ranked_topK_entity_idxs.indices.tolist()
            batch_head_entity = batch_head_entity.tolist()
            batch_correct_rate = np.sum(
                np.array([correct_rate(head_entity=head_entity, topK_entity_idx=ranked_topK_entity_idxs[idx],
                                       answers=batch_answers[idx], question=questions[idx], loader=qa_test_dataloader) for idx, head_entity in
                          enumerate(batch_head_entity)]))
            eval_correct_rate += batch_correct_rate
        eval_correct_rate /= (len(qa_test_dataloader) + len(qa_dev_dataloader))
        model.train()
        if eval_correct_rate > best_val_score + 0.0001:
            logger.info(
                f'evaluation accuracy hit@{TEST_TOP_K} increases from {best_val_score} to {eval_correct_rate}, save the model to {best_model_path}.')
            torch.save(model, best_model_path)
            best_val_score = eval_correct_rate
            NO_UPDATE = 0
        elif NO_UPDATE >= PATIENCE:
            logger.info(f'Model does not increases in the epoch-[{epoch_idx}], which has exceed patience.')
            exit(-1)
        else:
            NO_UPDATE += 1
logger.info(f"final epoch has reached. stop and save model to {final_model_path}.")
torch.save(model, final_model_path)
logger.info("bingo.")
