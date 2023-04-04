import os
import re
import nltk
import json
import tensorflow as tar

from agscripts import kgraph
from transformers import pipeline, set_seed
from transformers import BartTokenizer, BartForConditionalGeneration

nltk.download("stopwords")

AG_KMODEL_NAME = "./kmodels/K-files-cluster/"

class AgriBrainSuperPipeline(kgraph.KModel):
  def __init__(self, kmodel_path, K_summ_dir="./txt-clusters", seed=42):
    super(AgriBrainSuperPipeline, self).__init__(kmodel_path)
    self.k_summ_dir = K_summ_dir
    self.guidemodel_name = "gpt2-medium"
    self.aicore_name = "benkimz/agbrain"
    self.summ_lm_name = "philschmid/bart-large-cnn-samsum"
    self.seed = seed

    self.guidemodel_pipeline = pipeline('text-generation', 
                                        model=self.guidemodel_name)
    # self.aicore_pipeline = pipeline('text-generation', 
    #                                 model=self.aicore_name)
    self.summ_lm_tokenizer = BartTokenizer.from_pretrained(self.summ_lm_name)
    self.summ_lm_model=BartForConditionalGeneration.from_pretrained(self.summ_lm_name)
    set_seed(self.seed)

  def solve(self, prompt, max_length, num_return_sequences, temperature):
    cluster = self.get_cluster(prompt)
    summarized_context_guide = self.cluster_summary(
        self.k_summ_dir, cluster)
    mega_prompt = f"{summarized_context_guide} {prompt}"
    generated_guide = self.guidemodel_pipeline(
        mega_prompt, 
        num_return_sequences=num_return_sequences, 
        max_length=max_length, 
        num_beams=4, 
        no_repeat_ngram_size=2      
    )
    mega_context = []
    for guide in generated_guide:
      # to be replaced by aicore # code block at the bottom
      mega_context.append(guide["generated_text"])

    actual_response = []
    for data in mega_context:
      input_ids = self.summ_lm_tokenizer.encode(data, return_tensors="pt")
      summary_ids = self.summ_lm_model.generate(input_ids)
      actual_response.append(self.summ_lm_tokenizer.decode(summary_ids[0], 
                                                   skip_special_tokens=True))
    return ". ".join(actual_response)


# aicore_response = self.aicore_pipeline(
#     re.sub(mega_prompt, "", guide["generated_text"]), 
#     num_return_sequences=num_return_sequences, 
#     max_length=max_length, 
#     num_beams=4, 
#     no_repeat_ngram_size=2             
# ) 
# aicore_context = []
# for data in aicore_context:
#   aicore_context.append(aicore_response["generated_text"])
# aicore_context = ". ".join(aicore_context)            
# mega_context.append(f'{guide["generated_text"]}{aicore_context}')