const $ = (id) => document.getElementById(id);
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};

function card(it) {
  const el = document.createElement("div");
  el.className = "card";
  const img = document.createElement("img");
  img.loading = "lazy";
  img.src = `/api/image/${it.article_id}`;
  img.alt = it.name;
  const t = document.createElement("div");
  t.className = "t";
  const n = document.createElement("div");
  n.className = "n";
  n.textContent = it.name || it.type;
  const s = document.createElement("div");
  s.className = "s";
  s.textContent = it.score !== undefined ? `${(it.score * 100).toFixed(0)}% similar` : `${it.sales.toLocaleString()} sold`;
  t.append(n, s);
  el.append(img, t);
  el.onclick = () => openProduct(it.article_id);
  return el;
}

function fill(container, items) {
  container.replaceChildren(...items.map(card));
}

async function openProduct(id) {
  const { item, similar } = await api(`/api/product/${id}`);
  $("m-img").src = `/api/image/${id}`;
  const info = $("m-info");
  info.replaceChildren();
  const h = document.createElement("h2");
  h.textContent = item.name;
  info.append(h);
  for (const line of [item.type, item.category, item.colour, `${item.sales.toLocaleString()} sold`, `Article ${id}`]) {
    const p = document.createElement("p");
    p.textContent = line;
    info.append(p);
  }
  fill($("m-sim"), similar);
  $("modal").hidden = false;
  $("modal").scrollTop = 0;
}

$("close").onclick = () => ($("modal").hidden = true);
$("modal").onclick = (e) => { if (e.target.id === "modal") $("modal").hidden = true; };
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("modal").hidden = true; });

$("file").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  $("upload-section").hidden = false;
  $("preview").src = URL.createObjectURL(f);
  $("upload-results").replaceChildren();
  $("upload-msg").textContent = "Finding similar items… (the first upload loads the CLIP model and can take a while)";
  const fd = new FormData();
  fd.append("file", f);
  try {
    const { results } = await api("/api/search-by-image?k=15", { method: "POST", body: fd });
    fill($("upload-results"), results);
    $("upload-msg").textContent = "";
  } catch (err) {
    $("upload-msg").textContent = `Error: ${err.message}`;
  }
  e.target.value = "";
  $("upload-section").scrollIntoView({ behavior: "smooth" });
};

(async () => {
  fill($("best"), await api("/api/bestsellers?limit=14"));
  for (const { category, items } of await api("/api/bestsellers/by-category?limit=7")) {
    if (!items.length) continue;
    const row = document.createElement("div");
    row.className = "cat-row";
    const h = document.createElement("h3");
    h.textContent = category;
    const g = document.createElement("div");
    g.className = "grid";
    fill(g, items);
    row.append(h, g);
    $("by-cat").append(row);
  }
})();
