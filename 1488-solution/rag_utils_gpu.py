import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import sys
import os

LOCAL_EMB_DIR = os.path.join(os.path.dirname(__file__), "all-MiniLM-L6-v2")
META_PATH = "rag_meta_unique.jsonl"
EMB_PATH = "embeddings.npy"

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class RAG:
    def __init__(self, meta_path=META_PATH, emb_path=EMB_PATH, emb_model=EMB_MODEL, device=None, chunk_size=200000):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(self.device, str):
            self.torch_device = torch.device(self.device)
        else:
            self.torch_device = torch.device(str(self.device))
        print(f"[RAG] Device = {self.torch_device}", file=sys.stderr)

        model_source = emb_model
        if os.path.isdir(LOCAL_EMB_DIR):
            model_source = LOCAL_EMB_DIR
            print(f"[RAG] Found local sentence-transformers model at {LOCAL_EMB_DIR} -> loading from local files", file=sys.stderr)
        else:
            print(f"[RAG] Loading sentence-transformers model by name: {emb_model}", file=sys.stderr)

        try:

            self.model = SentenceTransformer(model_source, device=self.torch_device)

            print(f"[RAG] SentenceTransformer loaded from: {model_source}", file=sys.stderr)
        except Exception as e:
            print(f"[RAG] ERROR loading SentenceTransformer({model_source}): {e}", file=sys.stderr)

            try:
                self.model = SentenceTransformer(model_source, device=torch.device("cpu"))
                print("[RAG] Loaded model on CPU as fallback", file=sys.stderr)
                self.torch_device = torch.device("cpu")
            except Exception as e2:
                print(f"[RAG] FATAL: cannot load embedding model: {e2}", file=sys.stderr)
                self.model = None

        self.meta = []
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                for ln in f:
                    if not ln.strip(): continue
                    self.meta.append(json.loads(ln))
        except FileNotFoundError:
            print(f"[RAG] meta file not found: {meta_path}", file=sys.stderr)
        self.N = len(self.meta)
        if self.model is not None:
            try:
                self.dim = self.model.get_sentence_embedding_dimension()
            except Exception:

                try:
                    v = self.model.encode(["test"], convert_to_numpy=True, normalize_embeddings=False)
                    self.dim = v.shape[1]
                except Exception:
                    self.dim = None
        else:
            self.dim = None
        print(f"[RAG] Loaded meta entries: {self.N} dim={self.dim}", file=sys.stderr)

        if not os.path.exists(emb_path):
            print(f"[RAG] embeddings file not found: {emb_path}. You should run prepare_embeddings.py", file=sys.stderr)
            self.emb_memmap = None
            self.emb_gpu = None
            self.chunk_size = chunk_size
            return

        if self.dim is None:
            print("[RAG] WARNING: cannot determine embedding dim from model; loading memmap anyway (will assume shape based on file)", file=sys.stderr)

        self.emb_memmap = np.memmap(emb_path, dtype="float32", mode="r", shape=(self.N, self.dim))
        print("[RAG] embeddings memmap loaded", file=sys.stderr)

        self.emb_gpu = None
        if str(self.torch_device).startswith("cuda"):
            try:
                batch_size = 200_000
                emb_batches = []
                for start in range(0, self.N, batch_size):
                    end = min(self.N, start + batch_size)
                    batch = torch.from_numpy(np.asarray(self.emb_memmap[start:end])).to(self.torch_device)
                    emb_batches.append(batch)
                if emb_batches:
                    self.emb_gpu = torch.cat(emb_batches, dim=0)
                    print("[RAG] embeddings copied to GPU memory (batched)", file=sys.stderr)
            except Exception as e:
                print(f"[RAG] cannot copy embeddings to GPU (OOM?) -> will use chunked search: {e}", file=sys.stderr)
                self.emb_gpu = None

        self.chunk_size = chunk_size


    def embed(self, texts):
        if self.model is None:
            return np.zeros((len(texts), self.dim or 384), dtype="float32")
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        vecs = np.asarray(vecs, dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        return vecs

    def _topk_from_scores(self, scores_np, k):
        if k >= scores_np.size:
            idx = np.argsort(-scores_np)
            return scores_np[idx], idx
        part = np.argpartition(-scores_np, k-1)[:k]
        top_idx = part[np.argsort(-scores_np[part])]
        return scores_np[top_idx], top_idx

    def retrieve(self, query, top_k=5):
        if self.emb_memmap is None and self.emb_gpu is None:
            print("[RAG] No precomputed embeddings found — computing all on the fly (slow)", file=sys.stderr)
            texts = [m.get("content") or m.get("qa_pair", {}).get("question") for m in self.meta]
            all_embs = self.embed(texts)
            emb_tensor = torch.tensor(all_embs, device=self.device)
            qv = torch.tensor(self.embed([query])[0].astype("float32"), device=self.device).unsqueeze(1)
            scores = torch.matmul(emb_tensor, qv).squeeze(1)
            topk_scores, topk_idx = torch.topk(scores, k=min(top_k, scores.shape[0]))
            topk_idx = topk_idx.cpu().numpy().tolist()
            topk_scores = topk_scores.cpu().numpy().tolist()
            results = []
            for s, idx in zip(topk_scores, topk_idx):
                m = self.meta[idx]
                text = m.get("content") or m.get("qa_pair", {}).get("question")
                results.append({"score": float(s), "text": text, "meta": m})
            return results

        qv_np = self.embed([query])[0].astype("float32")

        if self.emb_gpu is not None:
            qv = torch.tensor(qv_np, device=self.device).unsqueeze(1)
            scores = torch.matmul(self.emb_gpu, qv).squeeze(1)
            topk = torch.topk(scores, k=min(top_k, scores.shape[0]))
            topk_scores = topk.values.cpu().numpy()
            topk_idx = topk.indices.cpu().numpy()
            results = []
            for s, idx in zip(topk_scores.tolist(), topk_idx.tolist()):
                m = self.meta[int(idx)]
                text = m.get("content") or m.get("qa_pair", {}).get("question")
                results.append({"score": float(s), "text": text, "meta": m})
            return results

        N = self.N
        k = min(top_k, N)
        best_scores = np.full(k, -np.inf, dtype="float32")
        best_idx = np.full(k, -1, dtype=np.int64)

        qv = qv_np
        for start in range(0, N, self.chunk_size):
            end = min(N, start + self.chunk_size)
            block = np.asarray(self.emb_memmap[start:end])
            sc = block.dot(qv)
            if sc.size == 0:
                continue

            local_k = min(k, sc.size)
            part = np.argpartition(-sc, local_k - 1)[:local_k]
            local_idx = part[np.argsort(-sc[part])]
            local_scores = sc[local_idx]

            cand_scores = np.concatenate([best_scores, local_scores])
            cand_idx = np.concatenate([best_idx, start + local_idx])
            sel = np.argpartition(-cand_scores, k-1)[:k]
            order = sel[np.argsort(-cand_scores[sel])]
            best_scores = cand_scores[order]
            best_idx = cand_idx[order]

        results = []
        for s, idx in zip(best_scores.tolist(), best_idx.tolist()):
            if int(idx) < 0:
                continue
            m = self.meta[int(idx)]
            text = m.get("content") or m.get("qa_pair", {}).get("question")
            results.append({"score": float(s), "text": text, "meta": m})
        return results

