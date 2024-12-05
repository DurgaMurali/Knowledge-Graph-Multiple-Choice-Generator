import time
from dataloader import MetaQADataLoader, DEV_MetaQADataLoader
from model import Answer_filtering_module
import torch
import logging
from tqdm import tqdm
import os
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
import networkx as nx
import knowledge_graph as kg
from scipy.spatial.distance import pdist
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hops", type=int, default="1", nargs="?",
                    help="Number of hops.")
    parser.add_argument("--model_path", type=str, default="./MetaQA_full_1_hop_singleAns_EntityTypeRecog/best_afm_model.pt", nargs="?",
                    help="Model Path.")

    args = parser.parse_args() 

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # =========
    KG_NAME = 'MetaQA'
    # =========
    HOPS = args.hops
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
    embedding_dim = entity_embeddings.shape[-1]
    relation_embeddings = qa_dataloader.dataset.relation_embeddings
    relation_dim = relation_embeddings.shape[-1]
    vocab_size = len(qa_dataloader.dataset.word_idx)
    word_dim = 256
    hidden_dim = 200
    fc_hidden_dim = 400

    model = torch.load(args.model_path)
    model.to(device)
    model.eval()

    TEST_TOP_K = 15

    epoch_test_process = tqdm(qa_test_dataloader, total=len(qa_test_dataloader), unit=' batches')
    undirected_KG = kg.KG.to_undirected()

    distances = pdist(entity_embeddings, metric='euclidean')
    min_distance = distances.min()
    max_distance = distances.max()


    def cosine_similarity(candid, answer):
        candidate_embed = torch.from_numpy(entity_embeddings[candid]).unsqueeze(0)
        answer_embed = torch.from_numpy(entity_embeddings[answer]).unsqueeze(0)
        cosine_sim = F.cosine_similarity(candidate_embed, answer_embed)
        return cosine_sim
    
    paths_from_head = 0
    no_paths = 0
    def path_from_head(candid, head, answer):
        try:
            path = nx.shortest_path(kg.KG, source=head, target=candid)
            global paths_from_head
            paths_from_head = paths_from_head + 1
            return len(path)
        except nx.NetworkXNoPath:
            print("No paths at all!")
            global no_paths
            no_paths = no_paths + 1
            return 0
            
    def euclidean_distance(candid, answer):
        candidate_embed = torch.from_numpy(entity_embeddings[candid]).unsqueeze(0)
        answer_embed = torch.from_numpy(entity_embeddings[answer.item()]).unsqueeze(0)
        return torch.norm(answer_embed - candidate_embed)


    insufficient_choices = 0
    average_cosine_sim = []
    average_choices_with_path = []
    average_choices_hops = []
    average_euclidean_distance = []

    # Filter candidates that contain the same entity type
    def filter_candidate_list(batch_head_entity, ranked_topK_entity_idxs, batch_answers, questions, loader):
        for idx, head_entity in enumerate(batch_head_entity):
            answers = batch_answers[idx]
            topK_entity_idx = ranked_topK_entity_idxs[idx]

            global insufficient_choices
            ans_entity_type = kg.ENTITY_TYPE_DICT[answers[0].item()]
            correct_choices = 0
            filtered_candid = []
            jaccard_pairs = []
            num_choices_with_path = 0
            for candid in topK_entity_idx:
                candid_entity_type = kg.ENTITY_TYPE_DICT[candid]
                if candid != answers[0].item() and candid != head_entity and any(type in ans_entity_type for type in candid_entity_type):
                    correct_choices = correct_choices+1
                    filtered_candid.append(candid)
            
            if correct_choices < 3:
                insufficient_choices = insufficient_choices+1
                print(questions[idx])
                print(loader.dataset.idx2entities[answers[0].item()])
                print(loader.dataset.idx2entities[topK_entity_idx[0]])
                print(loader.dataset.idx2entities[topK_entity_idx[1]])
                print(loader.dataset.idx2entities[topK_entity_idx[2]])
                print(loader.dataset.idx2entities[topK_entity_idx[3]])
                print("------------------------------------------")

            else:
                num_choices = 3
                filtered_candid = filtered_candid[:num_choices]
                cosine_sim = []
                num_hops = []
                euclidean_dist = []
                for candidate in filtered_candid:
                    sim = cosine_similarity(candidate, answers[0]).item()
                    cosine_sim.append([candid, loader.dataset.idx2entities[candid], sim])
                    
                    path_length = path_from_head(str(candidate), str(head_entity), str(answers[0].item()))
                    if path_length > 0:
                        num_choices_with_path += 1
                        num_hops.append(path_length-1)


                    euclidean_dist.append(euclidean_distance(candidate, answers[0]))

                similarities = [sublist[2] for sublist in cosine_sim]
                average_similarity = sum(similarities)/num_choices
                average_cosine_sim.append(average_similarity)

                average_choices_with_path.append(num_choices_with_path/num_choices)
                average_hops = sum(num_hops)/num_choices
                average_choices_hops.append(average_hops)

                average_dist = sum(euclidean_dist)/num_choices
                average_euclidean_distance.append(average_dist)


    eval_correct_rate = 0
    for questions, batch_questions_index, batch_questions_length, batch_head_entity, batch_answers, max_sent_len in epoch_test_process:
        ranked_topK_entity_idxs = model.get_ranked_top_k(batch_questions_index, batch_questions_length,
                                                            batch_head_entity, max_sent_len, K=TEST_TOP_K)
        ranked_topK_entity_idxs = ranked_topK_entity_idxs.indices.tolist()
        batch_head_entity = batch_head_entity.tolist()
        
        filter_candidate_list(batch_head_entity, ranked_topK_entity_idxs, batch_answers, questions, loader=qa_test_dataloader)
            
    print("insufficient_choices = " + str(insufficient_choices))
    print(len(average_cosine_sim), len(average_choices_with_path), len(average_choices_hops), len(average_euclidean_distance))
    print(sum(average_cosine_sim)/len(average_cosine_sim))
    print(paths_from_head, no_paths)
    print(sum(average_choices_with_path)/len(average_choices_with_path))
    print(sum(average_choices_hops)/len(average_choices_hops))
    print(min_distance, max_distance)
    print(sum(average_euclidean_distance)/len(average_euclidean_distance))