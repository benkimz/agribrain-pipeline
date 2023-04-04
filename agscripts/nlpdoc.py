import os
import spacy
import pickle

nlp = spacy.load('en_core_web_sm')

def load_nlp_doc(cluster_dir):
    assets_dir = os.path.join(cluster_dir, "components/")
    assert os.path.exists(assets_dir), "NLP doc not found"
    file_path = os.path.join(assets_dir, "nlp-doc.sav")
    assert os.path.exists(file_path), "NLP doc not found"
    return pickle.load(open(file_path, mode="rb"))
    
def preprocess_cluster_context(cluster_dir):
    txt_files = [os.path.join(cluster_dir, file) for file in os.listdir(cluster_dir) \
                 if os.path.join(cluster_dir, file).endswith(".txt")]
    if not len(txt_files) > 0: return
    context = []
    for file_path in txt_files:
        with open(file_path, mode="r", encoding="utf-8") as rtx:
            context.append(rtx.read())
    context = "\n\n".join(context)
    components_dir = os.path.join(cluster_dir, "components")
    if not os.path.exists(components_dir): os.mkdir(components_dir)
    nlp_tkz = nlp(context)
    pickle.dump(nlp_tkz, open(os.path.join(components_dir, "nlp-doc.sav"), mode="wb"))