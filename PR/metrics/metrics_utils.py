import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import yaml

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# check if synset category exists
def check_synset_exists(synset_name):
    try:
        # attempt to get synset using provided name
        syn = wn.synset(synset_name)
        return True, syn.definition() # TODO: do we need the 2nd output?
    except nltk.corpus.reader.wordnet.WordNetError:
        return False, None # TODO: do we need the 2nd output
    
# group classes by provided synset categories
def group_classes(labels, categories, class_to_classid):
    category_names = list(categories.keys())
    synset_names = list(categories.values())

    # check if synset is valid
    for synset_name in synset_names:
        if not check_synset_exists(synset_name):
            raise ValueError(
                f"The synset '{synset_name}' is not valid."
            )
    
    # initialize list to store category labels
    category_labels = []
    # loop through all labels
    for label in labels:
        # get class_id from the class_to_classid dictionary
        class_id = class_to_classid[label.item()]
        synset = wn.synset_from_pos_and_offset('n', int(class_id[1:]))
        searching = True

        while searching:
            # check if synset is in the synset_list
            if synset.name() in synset_names:
                # append category to category_labels
                category_labels.append(synset_names.index(synset.name()))
                searching = False
            else:
                try:
                    # get parent from hypernym tree
                    synset = synset.hypernyms()[0]
                except:
                    # if no more parents in tree, append -1 to category_labels list
                    category_labels.append(-1)
                    searching = False

    return category_names, category_labels
