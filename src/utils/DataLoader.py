import os
import collections
import smart_open
import random
import numpy as np

import Config


"""
Load basic ingredients and compounds data from Nature Scientific Report(Ahn, 2011)

"""
class DataLoader:
    # {ingredient_id: [ingredient_id1, ingredient_id2, ...] }
    def load_relations(self, path):
        relations = {}
        with open(path, 'r') as f:
            for line in f:
                if line[0] == '#':
                    pass
                else:
                    line_split = line.rstrip().split('\t')
                    ingredient_id = line_split[0]
                    compound_id = line_split[1]

                    if ingredient_id in relations:
                        relations[ingredient_id].append(compound_id)

                    else:
                        relations[ingredient_id] = [compound_id]

        return relations

    # {ingredient_id: [ingredient_name, ingredient_category]}
    def load_ingredients(self, path):
        ingredients = {}
        ingredients_list = []
        with open(path, 'r') as f:
            for line in f:
                if line[0] == '#':
                    pass
                else:
                    line_split = line.rstrip().split('\t')
                    ingredients_id = line_split[0]
                    ingredients_list = line_split[1:]
                    ingredients[ingredients_id] = ingredients_list
        return ingredients

    # {compound_id: [compound_name, CAS_number]}
    def load_compounds(self, path):
        compounds = {}
        compounds_list = []
        with open(path, 'r') as f:
            for line in f:
                if line[0] == '#':
                    pass
                else:
                    line_split = line.rstrip().split('\t')
                    compounds_id = line_split[0]
                    compounds_list = line_split[1:]
                    compounds[compounds_id] = compounds_list
        return compounds

    def batch_iter(self, data, batch_size):
        #data = np.array(data)
        data_size = len(data)
        num_batches = int(len(data)/batch_size)
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield data[start_index:end_index]

    def load_data(self, train, feat_dim):
        from nltk.stem import WordNetLemmatizer
        import GensimModels
        gensimLoader = GensimModels.GensimModels()
        model_loaded = gensimLoader.load_word2vec(path=Config.path_embeddings_ingredients)

        cult2id = {}
        id2cult = []
        comp2id = {'Nan':0}
        id2comp = ['Nan']

        train_cult = []
        train_comp = []
        train_comp_len = []

        comp_thr = 5
        max_comp_cnt = 0
        filtred_comp = 0

        train_f = open(train, 'r')
        lines = train_f.readlines()[4:]
        random.shuffle(lines)
        train_thr = int(len(lines) * 0.7)
        valid_thr = int(len(lines) * 0.8)

        print "Build composer dictionary..."
        for i, line in enumerate(lines):

            tokens = line.strip().split(',')
            culture = tokens[0]
            composers = tokens[1:]

            if cult2id.get(culture) is None:
                cult2id[culture] = len(cult2id)
                id2cult.append(culture)

            if comp_thr > len(composers):
                filtred_comp += 1
                continue

            if max_comp_cnt < len(composers):
                max_comp_cnt = len(composers)

            for composer in composers:
                if comp2id.get(composer) is None:
                    comp2id[composer] = len(comp2id)
                    id2comp.append(composer)

            train_cult.append(cult2id.get(culture))
            train_comp.append([comp2id.get(composer) for composer in composers])

        for comp in train_comp:
            train_comp_len.append(len(comp))
            if len(comp) < max_comp_cnt:
                comp += [0]*(max_comp_cnt - len(comp))

        wv = model_loaded.wv
        w = model_loaded.index2word
        #print [model_loaded[idx] for idx in w]
        wv_var = np.var([model_loaded[idx] for idx in w])

        '''
        compid2vec = np.array([np.random.rand(feat_dim) if comp not in wv 
                                                        else model_loaded[comp] for comp in id2comp])
        '''
        
        wnl = WordNetLemmatizer()
        mu, sigma = 0, 1
        compid2vec = []
        unk_cnt = 0
        for comp in id2comp:
            if comp in wv:
                compid2vec.append(model_loaded[comp])
            elif wnl.lemmatize(comp) in wv:
                compid2vec.append(model_loaded[wnl.lemmatize(comp)])
            elif comp.rstrip().split('_')[-1] in wv:
                compid2vec.append(model_loaded[comp.rstrip().split('_')[-1]])
            elif wnl.lemmatize(comp.rstrip().split('_')[-1]) in wv:
                compid2vec.append(model_loaded[wnl.lemmatize(comp.rstrip().split('_')[-1])])
            else:
                compid2vec.append(np.random.normal(mu, sigma, feat_dim))
                unk_cnt += 1

        print "unk cnt :", unk_cnt, "in", len(id2comp)
        print "filtered composer count is", filtred_comp

        return id2cult, id2comp, train_cult[:train_thr], train_comp[:train_thr], train_comp_len[:train_thr], train_cult[train_thr:valid_thr], train_comp[train_thr:valid_thr], train_comp_len[train_thr:valid_thr], train_cult[valid_thr:], train_comp[valid_thr:], train_comp_len[valid_thr:], max_comp_cnt, compid2vec

    # Ingredient_to_category
    def ingredient_to_category(self, tag, ingredients):
        for ingr_id in ingredients:
            if ingredients[ingr_id][0] == tag:
                return ingredients[ingr_id][1]
            else: 
                continue
        return


    # Corpus tag to index
    def tag_to_index(tags, corpus):
        for doc_id in range(len(corpus)):
            if tags == corpus[doc_id].tags[0]:
                return doc_id
            else:
                continue
        return

    # Corpus index to tag                    
    def index_to_tag(index, corpus):
        return corpus[index].tags


    # Cuisine - Ingredients
    def load_cultures(self, path):
        cultures = {}
        ingredient_list = []
        vocab = []

        with open(path, 'r') as f:
            culture_id = 0
            for culture_id, line in enumerate(f):
                if line[0] == '#':
                    pass
                else:
                    line_split = line.rstrip().split(',')
                    culture_label = line_split[0]
                    ingredient_list = line_split[1:]
                    cultures[culture_id] = [ingredient_list, [culture_label]]
                    for ingr in ingredient_list:
                        vocab.append(ingr)

        return cultures, set(vocab)

if __name__ == '__main__':
    dl = DataLoader()
    #ingredients = dl.load_ingredients(Config.path_ingr_info)
    #compounds = dl.load_compounds(Config.path_comp_info)
    #relations = dl.load_relations(Config.path_ingr_comp)

    cuisines = dl.load_cuisine(Config.path_cuisine)
