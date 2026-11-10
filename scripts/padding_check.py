import sys, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.slm import serialise, score
M = "models/Qwen__Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, device_map="cuda").eval()
f = pd.read_parquet("data/processed/features.parquet"); s = pd.read_parquet("data/processed/splits.parquet")
rows = f.merge(s, on="row_id").query("split == 'valid'").sample(64, random_state=1)
texts = [serialise.scoring_prompt(r) for _, r in rows.iterrows()]
p1, _ = score.score_texts(model, tok, texts, "cuda", 1, 9834, 902)
p8, _ = score.score_texts(model, tok, texts, "cuda", 8, 9834, 902)
lens = [len(tok(t + "Label:").input_ids) for t in texts]
print("rows", len(texts), "prompt len", min(lens), "-", max(lens), "| max |p1-p8| =", float(np.max(np.abs(np.array(p1) - np.array(p8)))))
import scipy.stats as st; d=np.abs(np.array(p1)-np.array(p8)); i=int(d.argmax()); print("worst row p1=%.4f p8=%.4f | spearman %.5f" % (p1[i], p8[i], st.spearmanr(p1,p8)[0]))
