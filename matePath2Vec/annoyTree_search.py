#encoding:utf-8

import sys
from annoy import AnnoyIndex
import pickle
reload(sys)
sys.setdefaultencoding('utf-8')	

class AnnoySearch(object):
    def __init__(self, input_file=None, 
                       model_path=None,
                       dict_path=None,
                       vec_dim=128, 
                       tree_num=10):
        self.input_file = input_file
        self.model_path = model_path
        self.dict_path = dict_path
        self.vec_dim = vec_dim
        self.tree_num = tree_num
        self._vecs_train = []
        self._ids = []
        self._id_index = dict()
        self._index_id = dict()

        self._annoy_tree = None
        self.__load()

    def __load(self):
        if self.input_file:
            with open(self.input_file, 'r') as f:
                for line in f:
                    arr = line.strip().split(' ')
                    id = arr[0]
                    vec = [float(sub) for sub in arr[1:]]
                    self._vecs_train.append(vec)
                    self._ids.append(id)
        if self.model_path and self.dict_path:
            self._annoy_tree = AnnoyIndex(self.vec_dim)
            self._annoy_tree.load(self.model_path)
            dict_file = open(self.dict_path, 'rb')
            dict_list = pickle.load(dict_file)
            self._id_index, self._index_id = dict_list

    def build_tree(self):
        self._annoy_tree = AnnoyIndex(self.vec_dim)
        for index, id in enumerate(self._ids):
            self._id_index[id] = index
            self._index_id[index] = id
        for index, vec_train in enumerate(self._vecs_train):
            #print vec_train
            self._annoy_tree.add_item(index, vec_train)
        self._annoy_tree.build(self.tree_num)


    def save_tree(self, model_path, dict_path):
        self._annoy_tree.save(model_path)
        dict_file = open(dict_path, 'wb')
        pickle.dump([self._id_index, self._index_id], dict_file)
        dict_file.close()

    def find_nns_by_id(self, id, n_items=40, search_k=-1, include_distances=False):
        index = self._id_index[id]
        #print index
        if self._annoy_tree and self._id_index:
            ids_found = []
            res_found = self._annoy_tree.get_nns_by_item(index, n_items, search_k=search_k, include_distances=include_distances)
            #print res_found
            if include_distances:
                for index, dist in zip(res_found[0], res_found[1]):
                    id_found = self._index_id[index]
                    res = (id_found, dist)
                    ids_found.append(res)
            else:
                for index in res_found:
                    id_found = self._index_id[index]
                    ids_found.append(id_found)
        return ids_found

    def print_nns_by_file(self, id_file, n_items, search_k=-1, include_distances=False):
        for line in open(id_file, 'r').readlines():
            id = line.strip()
            index = self._id_index[id]
            res_found = self._annoy_tree.get_nns_by_item(index, n_items, search_k=search_k, include_distances=include_distances)
            #ids_found = self._index_id[index_found]
            
            if include_distances:
                for index, dist in zip(res_found[0], res_found[1]):
                    id_found = self._index_id[index]
                    print '%s\t%s\t%s' % (id, id_found, str(dist))
            else:
                for index in res_found:
                    id_found = self._index_id[index]
                    print '%s\t%s' % (id, id_found) 

if __name__ == "__main__":
    #id = sys.argv[1]
    ann_path = sys.argv[4]
    pkl_path = sys.argv[5]
    
    ans = AnnoySearch(model_path=ann_path, \
			          dict_path=pkl_path, \
					  vec_dim=100)

    input_file = sys.argv[1]
    K_top = int(sys.argv[2])
    dist_threshold = float(sys.argv[3])

    with open(input_file, 'r') as f:
        for line in f:
            arr = line.strip('\n').split('\t')
            nid = arr[1]
            id = 'i' + str(nid)
            #id = arr[0]
            try:
                ids_found = ans.find_nns_by_id(id, K_top, -1, True)
                for id_found, dist in ids_found:
                    if float(dist) <= dist_threshold and float(dist) >= 0.01:
                        print line.strip() + '\t' + id_found.strip('i') + '\t' + str(dist)
            except Exception as e:
                continue
