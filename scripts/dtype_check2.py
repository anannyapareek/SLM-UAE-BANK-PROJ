import time, numpy as np, pandas as pd, torch, scipy.stats as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.slm import serialise, score
M = "models/Qwen__Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
f = pd.read_parquet("data/processed/features.parquet"); s = pd.read_parquet("data/processed/splits.parquet")
texts = [serialise.scoring_prompt(r) for _, r in f.merge(s, on="row_id").query("split == 'valid'").sample(1000, random_state=1).iterrows()]
for name, dt, tf32 in (("fp16", torch.float16, False), ("fp32+tf32", torch.float32, True)):
    torch.backends.cuda.matmul.allow_tf32 = tf32; torch.backends.cudnn.allow_tf32 = tf32
    model = AutoModelForCausalLM.from_pretrained(M, dtype=dt, device_map="cuda").eval()
    p1, _ = score.score_texts(model, tok, texts[:64], "cuda", 1, 9834, 902)
    p8, _ = score.score_texts(model, tok, texts[:64], "cuda", 8, 9834, 902)
    torch.cuda.synchronize(); t = time.perf_counter()
    score.score_texts(model, tok, texts, "cuda", 16, 9834, 902)
    torch.cuda.synchronize(); r = 1000 / (time.perf_counter() - t)
    p1, p8 = np.array(p1), np.array(p8)
    print("%-10s %.1f rows/s | max diff %.2e | spearman %.5f | nan %d" % (name, r, np.abs(p1-p8).max(), st.spearmanr(p1, p8)[0], np.isnan(p1).sum()))
    del model; torch.cuda.empty_cache()
