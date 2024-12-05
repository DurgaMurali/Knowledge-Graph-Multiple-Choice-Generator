import networkx
import numpy as np

KG_NAME = 'MetaQA'
KG_HALF = False
BEST_OR_FINAL = 'best'

# Create KG
KG_EMBED_PATH = f'../knowledge_graph_embedding_module/kg_embeddings/{KG_NAME}{"_half" if KG_HALF else ""}/' \
                f'{BEST_OR_FINAL}_checkpoint/%s'
E_path = KG_EMBED_PATH % 'E.npy'
RELATION_EMBEDDINGS_PATH = KG_EMBED_PATH % 'R.npy'
RELATION_EMBEDDINGS = np.load(RELATION_EMBEDDINGS_PATH)
ENTITY_DICT_PATH = KG_EMBED_PATH % 'entities_idx.dict'
RELATION_DICT_PATH = KG_EMBED_PATH % 'relations_idx.dict'
ENTITY_DICT = dict()
with open(ENTITY_DICT_PATH, mode='rt', encoding='utf-8') as inp:
    for line in inp:
        split_infos = line.strip().split('\t')
        ENTITY_DICT[split_infos[0]] = split_infos[1]
RELATION_DICT = dict()
with open(RELATION_DICT_PATH, mode='rt', encoding='utf-8') as inp:
    for line in inp:
        split_infos = line.strip().split('\t')
        RELATION_DICT[split_infos[0]] = split_infos[1]

KG_GRAPH_PATH = f'../knowledge_graph_embedding_module/knowledge_graphs/{KG_NAME}{"_half" if KG_HALF else ""}/' \
                f'%s.txt'


def load_data(data_path, reverse):
    with open(data_path, 'r', encoding='utf-8') as inp_:
        data = [l.strip().split('\t') for l in inp_.readlines()]
        if reverse:
            data += [[i[2], i[1] + "_reverse", i[0]] for i in data]
        new_data = []
        for i in data:
            new_data.append([(ENTITY_DICT[i[0]], {'name': i[0]}), {'r_idx': RELATION_DICT[i[1]], 'r': i[1]},
                             (ENTITY_DICT[i[2]], {'name': i[2]})])
    return new_data


all_KG_triples = load_data(KG_GRAPH_PATH % 'train', True) + load_data(KG_GRAPH_PATH % 'valid', True) + load_data(
    KG_GRAPH_PATH % 'test', True)

KG = networkx.DiGraph()

# Add head entity from triple as node
KG.add_nodes_from(map(lambda x: x[0], all_KG_triples))  # store all nodes and their 'name' attributes.
# Add tail entity from triple as node
KG.add_nodes_from(map(lambda x: x[2], all_KG_triples))  # store all nodes and their 'name' attributes.
KG.add_edges_from(map(lambda x: (x[0][0], x[2][0], x[1]), all_KG_triples))  # store all edges and edges` type [r_idx, r]
print("Number of edges: " + str(KG.number_of_edges()) + " Number of nodes: " + str(KG.number_of_nodes()))

# 0 - People
# 1 - Genre
# 2 - Rating
# 3 - Votes
# 4 - Tags
# 5 - Language
# 6 - Year
# 7 - Movies/Books
ENTITY_REL_DICT = dict()
ENTITY_REL_DICT[0] = 0
ENTITY_REL_DICT[1] = 7
ENTITY_REL_DICT[2] = 1
ENTITY_REL_DICT[3] = 7
ENTITY_REL_DICT[4] = 2
ENTITY_REL_DICT[5] = 7
ENTITY_REL_DICT[6] = 3
ENTITY_REL_DICT[7] = 7
ENTITY_REL_DICT[8] = 4
ENTITY_REL_DICT[9] = 7
ENTITY_REL_DICT[10] = 5
ENTITY_REL_DICT[11] = 7
ENTITY_REL_DICT[12] = 6
ENTITY_REL_DICT[13] = 7
ENTITY_REL_DICT[14] = 0
ENTITY_REL_DICT[15] = 7
ENTITY_REL_DICT[16] = 0
ENTITY_REL_DICT[17] = 7

ENTITY_TYPE_DIST = [0,0,0,0,0,0,0,0]
ENTITY_TYPE_DICT = dict()
for node in KG.nodes:
    inc_edges = KG.in_edges(node, data=True)
    entity_type_set = set()
    for u, v, attr in inc_edges:
        rel_type = int(attr['r_idx'])
        entity_type_set.add(ENTITY_REL_DICT[rel_type])
    ENTITY_TYPE_DICT[int(node)] = list(entity_type_set)