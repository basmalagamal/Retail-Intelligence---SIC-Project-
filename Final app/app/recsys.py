import io
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
EMB_DIR = Path(os.getenv("EMB_DIR", ROOT))
CACHE = ROOT / "cache"
CATEGORY_COL = "product_group_name"


def locate(name):
    for root, _, files in os.walk(DATA_DIR):
        if name in files:
            return Path(root) / name
    return None


class ClipEmbedder:
    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __call__(self, pil_image):
        x = self.proc(images=pil_image.convert("RGB"), return_tensors="pt")["pixel_values"]
        with self.torch.no_grad():
            out = self.model.vision_model(pixel_values=x)
            return self.model.visual_projection(out.pooler_output)[0].numpy()


class RecSys:
    def __init__(self):
        emb = np.load(EMB_DIR / "emb_clip.npy")
        self.ids = pd.read_csv(EMB_DIR / "ids_resnet.csv", dtype={"article_id": str})["article_id"].tolist()
        assert len(self.ids) == len(emb), "ids / embedding length mismatch"
        self.emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype(np.float32)
        self.id2row = {a: r for r, a in enumerate(self.ids)}

        path = locate("articles_sampled.csv")
        if path is None:
            raise FileNotFoundError(f"articles_sampled.csv not found under {DATA_DIR} - run scripts/fetch_data.py")
        art = pd.read_csv(path, dtype={"article_id": str}).drop_duplicates("article_id")
        self.meta = art.set_index("article_id").reindex(self.ids)
        self.meta[CATEGORY_COL] = self.meta[CATEGORY_COL].fillna("Unknown")

        self.sales = self._sales_counts()
        self.sales_vec = self.sales.reindex(self.ids).fillna(0).astype(int).to_numpy()

        self.embedder = None

    def _sales_counts(self):
        CACHE.mkdir(exist_ok=True)
        cache = CACHE / "sales_counts.csv"
        tx = locate("transactions_sampled.csv")
        if cache.exists() and (tx is None or cache.stat().st_mtime >= tx.stat().st_mtime):
            return pd.read_csv(cache, dtype={"article_id": str}).set_index("article_id")["n"]
        if tx is None:
            raise FileNotFoundError(f"transactions_sampled.csv not found under {DATA_DIR}")
        total = pd.Series(dtype="int64")
        for chunk in pd.read_csv(tx, usecols=["article_id"], dtype={"article_id": str}, chunksize=1_000_000):
            total = total.add(chunk["article_id"].str.strip().str.zfill(10).value_counts(), fill_value=0)
        total = total.astype(int).rename("n").rename_axis("article_id")
        total.to_csv(cache)
        return total

    def image_path(self, aid):
        for p in (DATA_DIR / "images" / aid[:3] / f"{aid}.jpg",
                  DATA_DIR / "sampled_images" / f"{aid}.jpg"):
            if p.exists():
                return p
        for root in ("images", "sampled_images"):
            for p in (DATA_DIR / root).glob(f"**/{aid}.jpg"):
                return p
        return None

    def item(self, row, score=None):
        m = self.meta.iloc[row]
        d = {
            "article_id": self.ids[row],
            "name": str(m.get("prod_name", "")),
            "type": str(m.get("product_type_name", "")),
            "category": str(m[CATEGORY_COL]),
            "colour": str(m.get("colour_group_name", "")),
            "sales": int(self.sales_vec[row]),
        }
        if score is not None:
            d["score"] = round(float(score), 4)
        return d

    def categories(self):
        vc = self.meta[CATEGORY_COL].value_counts()
        return [{"name": k, "articles": int(v)} for k, v in vc.items()]

    def bestsellers(self, category=None, k=12):
        mask = np.ones(len(self.ids), bool) if not category else (self.meta[CATEGORY_COL] == category).to_numpy()
        rows = np.flatnonzero(mask)
        rows = rows[np.argsort(-self.sales_vec[rows], kind="stable")][:k]
        return [self.item(r) for r in rows if self.sales_vec[r] > 0]

    def top_similar(self, vec, k, exclude=None):
        sim = self.emb @ vec  # cosine, both sides unit-norm
        if exclude is not None:
            sim[exclude] = -np.inf
        top = np.argpartition(-sim, k)[:k]
        top = top[np.argsort(-sim[top])]
        return [self.item(r, sim[r]) for r in top]

    def similar(self, aid, k=12):
        r = self.id2row[aid]
        return self.top_similar(self.emb[r], k, exclude=r)

    def similar_to_image(self, data: bytes, k=12):
        from PIL import Image

        if self.embedder is None:
            self.embedder = ClipEmbedder()
        vec = np.asarray(self.embedder(Image.open(io.BytesIO(data))), dtype=np.float32)
        vec /= np.linalg.norm(vec)
        return self.top_similar(vec, k)
