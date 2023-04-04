import os
import re
import glob
import json
import tqdm
import spacy
import spacy.cli
import string
import pickle
import numpy as np
import tensorflow as tf

import nltk
nltk.data.path.append("/usr/share/nltk_data")

from textblob import TextBlob
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

spacy.cli.download("en")
spacy.cli.download("en_core_web_sm")

nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))  

def load_kmodel(model_name):
    assert os.path.exists(model_name), "Model does not exist"
    model_path = os.path.join(model_name, "kmeans-model.sav")
    kmeans_model = pickle.load(open(model_path, mode="rb"))
    return kmeans_model

def load_kvec(model_name):
    assert os.path.exists(model_name), "Model does not exist"
    vectorizer_path = os.path.join(model_name, "text-vectorizer.sav")
    vectorizer = pickle.load(open(vectorizer_path, mode="rb"))
    return vectorizer

class KModel:
    def __init__(self, model_name):
        assert (type(model_name) == str), "Pass a valid model_name"
        self.model_name = model_name
        self.kmodel = self.load_kmodel()
        self.kvec = self.load_kvec()

    def load_kmodel(self):
        assert os.path.exists(self.model_name), "Model does not exist"
        model_path = os.path.join(self.model_name, "kmeans-model.sav")
        return pickle.load(open(model_path, mode="rb"))

    def load_kvec(self):
        assert os.path.exists(self.model_name), "Model does not exist"
        vectorizer_path = os.path.join(self.model_name, "text-vectorizer.sav")
        return pickle.load(open(vectorizer_path, mode="rb"))   
        
    def get_cluster(self, inputs):
        encodings = self.kvec.transform([inputs])
        outputs = self.kmodel.predict(encodings)
        return outputs[0]

    def get_context(self, source_dir, cluster):
        target_dir = os.path.join(source_dir, f"cluster-{cluster}")
        assert os.path.exists(target_dir), "Cluster not found"
        num_txt_files = len(os.listdir(target_dir))
        if not num_txt_files > 0: return ''
        content = []
        for filename in os.listdir(target_dir):
            file_path = os.path.join(target_dir, filename)
            with open(file_path, mode="r", encoding="utf-8") as fp:
                content.append(fp.read())

        return "\n\n".join(content)
    
    def cluster_summary(self, source_dir, cluster):
        target_dir = os.path.join(source_dir, f"cluster-{cluster}")
        assert os.path.exists(target_dir), "Cluster not found"
        summary_file = os.path.join(target_dir, "summary/summary.txt")
        if not os.path.exists(summary_file): return ''
    
        with open(summary_file, mode="r", encoding="utf-8") as fp:
            content = fp.read()
            
        return content
    
    
    def extract_context(self, prompt, context, n_paras=2):
        # Preprocessing
        doc_prompt = nlp(prompt)

        prompt_tokens = [token.lemma_ for token in doc_prompt if not token.is_stop]
        # Split the context into paragraphs
        context_paragraphs = context.split("\n\n")

        # Process each paragraph separately
        paragraph_scores = []
        for paragraph in context_paragraphs:
            try:
                doc_context = nlp(paragraph)      
                context_tokens = [token.lemma_ for token in doc_context if not token.is_stop]
                if not len(context_tokens) > 0: continue
                max_n = 3
                n_grams = []
                for n in range(1, max_n+1):
                    n_grams += [doc_prompt[start:start+n].text.lower() \
                                for start in range(len(doc_prompt)-n+1)]
                    n_grams += [doc_context[start:start+n].text.lower() \
                                for start in range(len(doc_context)-n+1)]
                vectorizer = TfidfVectorizer()
                tfidf_context = vectorizer.fit_transform(context_tokens)

                # Handle grammatical errors
                prompt_corrected = ' '.join([self.spellcheck(token) for token in prompt_tokens])
                doc_prompt_corrected = nlp(prompt_corrected)
                prompt_tokens_corrected = [token.lemma_ for token in \
                                           doc_prompt_corrected if not token.is_stop]
                n_grams_corrected = []
                for n in range(1, max_n+1):
                    n_grams_corrected += [doc_prompt_corrected[start:start+n].text.lower() \
                                          for start in range(len(doc_prompt_corrected)-n+1)]

                # Find related context
                similarities = {}
                for n_gram in n_grams_corrected:
                    if n_gram not in vectorizer.vocabulary_:
                        continue
                    tfidf_prompt = vectorizer.transform([n_gram])
                    similarity_scores = cosine_similarity(tfidf_prompt, tfidf_context)[0]
                    if similarity_scores.max() > 0.5: # Threshold to adjust the amount of related context returned
                        similarities[n_gram] = similarity_scores.max()
                if similarities:
                    # Calculate the paragraph score as the average similarity score of the n-grams
                    paragraph_score = sum(similarities.values()) / len(similarities)
                    paragraph_scores.append((paragraph, paragraph_score))
                    
            except Exception as error: 
                continue
        # Sort the paragraphs by score and return the top n paragraphs
        top_paragraphs = [p[0] for p in sorted(paragraph_scores, key=lambda x: x[1], reverse=True)[:n_paras]]
        return top_paragraphs


    def spellcheck(self, token):
        if token in string.punctuation or token in stop_words:
            return token
        if wordnet.synsets(token):
            return token
        else:
            suggestions = TextBlob(token).correct().split()
            if len(suggestions) > 0:
                return suggestions[0]
            else:
                return token    
        
class AgBrainWheel(KModel):
    def __init__(self, kmodel_name, agbrain_model, agtokenizer, 
                 max_length=512, clusters_dir="./txt-clusters/"):
        assert type(kmodel_name) == str, "Invalid name for kmodel"
        super(AgBrainWheel, self).__init__(kmodel_name)
        self.model = agbrain_model
        self.tokenizer = agtokenizer
        self.max_length = max_length
        self.clusters_dir = clusters_dir
        return None
    
    def prompt_reply(self, prompt, draw_context=True, num_sequences=1):
        if draw_context:
            prompt, context = self.prompt_context(prompt)
        else: context = ""
        self.prompt = prompt
        self.context = context
        attention_mask, input_ids = self.prompt_context_encodings(prompt, context)
        return self.generate_response(input_ids, attention_mask, num_sequences)
    
    def prompt_context(self, prompt):
        cluster = self.get_cluster(inputs=prompt[:min(len(prompt), \
                                                      self.max_length)])
        K_summary = self.cluster_summary(source_dir=self.clusters_dir, \
                                         cluster=cluster)
        return prompt, K_summary
    
    def prompt_context_encodings(self, prompt, context):
        inputs = f"{context}{self.tokenizer.sep_token}{prompt}"
        tokenized = self.tokenizer(inputs)
        input_ids = tf.constant([tokenized["input_ids"]])
        attention_mask = tf.constant([tokenized["attention_mask"]])
        
        return attention_mask, input_ids
    
    def generate_response(self, input_ids, attention_mask, num_sequences):
        outputs = self.model.generate(
                input_ids=input_ids,  
                attention_mask=attention_mask, 
                top_k=50, 
                top_p=0.95, 
                temperature=1.0, 
                do_sample=True, 
                max_length=self.max_length,   
                no_repeat_ngram_size=3, 
                num_beams=2, 
                num_return_sequences=num_sequences, 
                early_stopping=True, 
                length_penalty=1.0, 
                repetition_penalty=1.0)
        text_output = []
        for output in outputs:
            text_output.append(self.tokenizer.decode(output, 
                                                skip_special_tokens=True))
        generated = " ".join(text_output)
        generated = re.sub(self.context, " ", generated)
        generated = re.sub(self.prompt, " ", generated)
        generated = re.sub(self.tokenizer.sep_token, " ", generated)
        generated = re.sub(r"\s+", " ", generated)
        generated = re.sub(r"\[\s+]+", "", generated)
        generated = re.sub(r"\[\]+", "", generated)
        return generated

class TextFilesParser:
    def __init__(self, source_dir=None, output_dir=None, extensions=[".txt"]):
        output_dir = output_dir if output_dir else "./txt-parser-ouputs/"
        assert type(output_dir) == str
        output_dir = re.sub(r"\s+", "_", output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if source_dir:
            for file in tqdm.tqdm(os.listdir(source_dir)):
                file_path = os.path.join(source_dir, os.path.basename(file))
                if not os.path.isfile(file_path): continue
                file_name, fext = os.path.splitext(file)
                if not fext in extensions: continue
                output_file = os.path.join(output_dir, file_name + fext)
                with open(file_path, mode="r", encoding="utf-8") as context:
                    content = context.read()
                    
                parser_content = self.paragraph_parser(content)
                if not parser_content is None and type(parser_content) == str:
                    with open(output_file, mode="w", encoding="utf-8") as fp:
                        fp.write(parser_content)
                        fp.close()
        return None
    
    def line_parser(self, line):
        line = re.sub(r"\s+", " ", line)
        line = re.sub(r"\t+", " ", line)
        line = re.sub(r"www?(?:[-\w.]|(?:%[\da-fA-F]{2}))+", '[URL_ADDRESS]', line)
        line = re.sub(r'^[\w-]+(\.[\w-]+)*@[\w-]+(\.[\w-]+)*(\.[a-zA-Z]{2,})$', 
                      '[EMAIL_ADDRESS]', line)
        if len(line.split("@")) >=2: return None
        if line and not line == " ":
            return line
        
    def paragraph_parser(self, corpus):
        # removing any non utf-8 characters
        non_utf8 = pattern = re.compile(r'[^\x00-\x7F]')
        corpus = non_utf8.sub("", corpus)

        # split the corpus into paragraphs
        paragraphs = corpus.split("\n\n")
        new_paragraphs = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            sents = [self.line_parser(line) for line in paragraph.split("\n") if self.line_parser(line)]       
            
            num_sents = len(sents)
            if not num_sents > 8:
                continue

            average_num_words_track = []
            for sent in sents:
                average_num_words_track.append(len(sent.split()))
                
            # Average words in a sentence
            average_words = np.mean(np.array(average_num_words_track))
            
            # Average symbols stops in a sentence
            symbols_list = [".", ",", ":", ";"]
            average_symbols_count = 0
            for symb in symbols_list:
                average_symbols_count += len(paragraph.split(symb)) / num_sents
            if average_symbols_count > 0:
                if (average_symbols_count / len(symbols_list)) >= 3:
                    continue
                    
            if not int(average_words) >= 8:
                continue
            paragraph = "".join(sents)
            paragraph = re.sub(r"www?(?:[-\w.]|(?:%[\da-fA-F]{2}))+", 
                               '[URL_ADDRESS]', paragraph)
            paragraph = re.sub(r'^[\w-]+(\.[\w-]+)*@[\w-]+(\.[\w-]+)*(\.[a-zA-Z]{2,})$', 
                          '[EMAIL_ADDRESS]', paragraph)            
            new_paragraphs.append(paragraph)
        if len(new_paragraphs) > 0:
            return "\n\n".join(new_paragraphs)       
        
class TextFileClusters(KMeans):
    def __init__(self, source_dir=None, output_dir=None, n_clusters=5, random_state=42, max_iter=1000000):
        
        output_dir = output_dir if output_dir else "./txt-cluster-output/"
        assert type(output_dir) == str
        self.output_dir = re.sub(r"\s+", "_", output_dir)
        
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.source_dir = source_dir
        
        assert os.path.exists(self.source_dir), "self.source_dir not found"
        
        self.files_kmeans = KMeans(n_clusters=self.n_clusters, 
                                   random_state=self.random_state, 
                                   max_iter=self.max_iter)
        
        self.files_text_vectorizer = TfidfVectorizer(stop_words="english")
        
        data = self._get_text_data()
        lbls = self._kmeans_clustering(data) ###
        self._save_clusters(lbls)
        
        return None
    
    def save_components(self, save_dir: str, overwrite_dir=False):
        if os.path.exists(save_dir) and not overwrite_dir:
            raise Exception("""
            K-model: {} exists. Select a different name or set to overwrite contents.
            """.format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, "kmeans-model.sav")
        vectorizer_path = os.path.join(save_dir, "text-vectorizer.sav")
        pickle.dump(self.files_kmeans, open(model_path, mode="wb"))
        pickle.dump(self.files_text_vectorizer, open(vectorizer_path, mode="wb"))
        print("Model, {} saved successfully.".format(save_dir))
    
    def _get_text_data(self):
        textdata = []
        for filepath in glob.glob(os.path.join(self.source_dir, "*.txt")):
            with open(filepath, mode="r", encoding="utf-8") as context:
                textdata.append(context.read())
        return textdata
    
    def _kmeans_clustering(self, text_data):
        # Perform k-means clustering on text data
        inputs = self.files_text_vectorizer.fit_transform(text_data)
        
        self.files_kmeans.fit(inputs)
        
        cluster_labels = self.files_kmeans.labels_
        
        return cluster_labels    
    
    def _save_clusters(self, cluster_labels):
        # Copy files to cluster directories
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        for i in range(max(cluster_labels) + 1):            
            cluster_dir = os.path.join(self.output_dir, f"cluster-{i}")
            if not os.path.exists(cluster_dir):
                os.mkdir(cluster_dir)
            for j, filename in enumerate(glob.glob(os.path.join(self.source_dir, '*.txt'))):
                if cluster_labels[j] == i:
                    outfile = os.path.join(cluster_dir, os.path.basename(filename))
                    with open(filename, mode="r", encoding="utf-8") as readctx, \
                    open(outfile, mode="w", encoding="utf-8") as writectx:
                        writectx.write(readctx.read()) 
            
# outdone class
class AgBrainManualWheel(KModel):
    def __init__(self, kmodel_name, agbrain_model, agtokenizer, 
                 max_length=512, max_context_length=2048, clusters_dir="./txt-clusters/"):
        assert type(kmodel_name) == str, "Invalid name for kmodel"
        super(AgBrainWheel, self).__init__(kmodel_name)
        self.model = agbrain_model
        self.tokenizer = agtokenizer
        self.max_length = max_length
        self.max_context_length = max_context_length
        self.clusters_dir = clusters_dir
        return None
    
    def prompt_reply(self, prompt):
        prompt, context = self.prompt_context(prompt)
        self.prompt = prompt
        self.context = context
        model_inputs = self.prompt_context_ids(prompt, context)
        return self.generate_response(model_inputs)
    
    def generate_response(self, input_ids):
        outputs = model.generate(
                input_ids=input_ids,  
                #attention_mask=attention_mask, 
                top_k=50, 
                top_p=0.95, 
                temperature=1.0, 
                do_sample=True, 
                max_length=self.max_length,   
                no_repeat_ngram_size=3, 
                num_beams=4, 
                num_return_sequences=2, 
                early_stopping=True, 
                length_penalty=1.0, 
                repetition_penalty=1.0)
        text_output = []
        for output in outputs:
            text_output.append(tokenizer.decode(output, 
                                                skip_special_tokens=True))
        generated = " ".join(text_output)
        generated = re.sub(self.context, " ", generated)
        generated = re.sub(self.prompt, " ", generated)
        return generated
    
    def prompt_context(self, prompt):
        cluster = self.get_cluster(inputs=prompt[:min(len(prompt), self.max_length)])
        raw_content = self.get_context(source_dir=self.clusters_dir, cluster=cluster)
        context = "\n\n".join(self.extract_context(prompt, raw_content, n_paras=2))
        return prompt, context[:min(len(context), self.max_context_length)]
    
    def get_context_segments(self, context):
        context = context[:min(len(context), self.max_context_length)]
        return [context[i : i + self.max_length] \
                for i in range(0, len(context), self.max_length)]
    
    def prompt_context_ids(self, prompt, context):
        prompt_ids = self.get_prompt_ids(prompt)
        context_segments = self.get_context_segments(context)
        context_ids = self.get_context_ids(context_segments)
        return tf.concat([context_ids, prompt_ids], axis=-1)   
    
    def get_context_ids(self, context_segments):
        input_ids = []
        for segment in context_segments:
            encoded_segment = self.tokenizer.encode(segment,
                                        truncation=True,  
                                        max_length=MAX_LEN, 
                                        return_tensors="tf")
            input_ids.append(encoded_segment)
        return tf.concat(input_ids, axis=-1)

    def get_prompt_ids(self, prompt):
        prompt_ids = tokenizer.encode(prompt, 
                               truncation=True, 
                               max_length=MAX_LEN, 
                               return_tensors="tf")
        return prompt_ids                
