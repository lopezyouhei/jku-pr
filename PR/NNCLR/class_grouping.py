import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

# just a function to help with checking if category exists
def check_synset_exists(synset_name):
    try:
        # Attempt to fetch the synset using the provided name
        syn = wn.synset(synset_name)
        return True, syn.definition()  # Return True and the definition if it exists
    except nltk.corpus.reader.wordnet.WordNetError:
        return False, None  # Return False and None if the synset does not exist

def get_category(labels, categories, class_to_classid):
    category_names = list(categories.keys())
    synset_list = list(categories.values())
    
    for synset_name in synset_list:
        if not check_synset_exists(synset_name):
            raise ValueError(
                f"The synset '{synset_name}' is not valid."
                )
    
    # initialize list to store category labels
    category_labels = []
    # loop through all labels
    for label in labels:
        # get the class_id from the class_to_classid dictionary
        class_id = class_to_classid[label.item()]
        synset = wn.synset_from_pos_and_offset('n', int(class_id[1:]))
        searching = True
        while searching:
            # check if the synset is in the synset_list
            if synset.name() in synset_list:
                # if it is, add the category to the category_labels list
                category_labels.append(synset_list.index(synset.name()))
                searching = False
            else:
                try:
                    # if it is not, move up the hypernym tree
                    synset = synset.hypernyms()[0]
                except:
                    # if there are no more hypernyms, add -1 to the category_labels list
                    category_labels.append(-1)
                    searching = False

    return category_names, category_labels