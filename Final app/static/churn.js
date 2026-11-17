const pct = (x) => `${(x * 100).toFixed(1)}%`;
const colour = { high: "#c8102e", medium: "#e8a317", low: "#2e9e5b" };
let churnLoaded = false;

function bars(el, rows, cls) {
  el.replaceChildren();
  const max = Math.max(...rows.map((r) => r.value), 1e-9);
  for (const r of rows) {
    const d = document.createElement("div"); d.className = "bar";
    const l = document.createElement("span"); l.textContent = r.label;
    const w = document.createElement("div"); const i = document.createElement("i");
    i.style.width = `${(r.value / max) * 100}%`; if (cls || r.cls) i.className = r.cls || cls; w.append(i);
    const v = document.createElement("span"); v.textContent = r.text;
    d.append(l, w, v); el.append(d);
  }
}

function showView(v) {
  $("shop-view").hidden = v !== "shop"; $("churn-view").hidden = v !== "churn"; $("upload-btn").hidden = v !== "shop";
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
  if (v === "churn" && !churnLoaded) loadChurn();
}
document.querySelectorAll(".tab").forEach((b) => (b.onclick = () => showView(b.dataset.view)));

async function loadChurn() {
  $("churn-error").textContent = "Scoring customers… (first load can take a moment)";
  let s;
  try { s = await api("/api/churn/summary"); } catch (e) { $("churn-error").textContent = `Churn module unavailable: ${e.message}`; return; }
  churnLoaded = true; $("churn-error").textContent = ""; $("churn-body").hidden = false;
  const kp = [[s.scored.toLocaleString(), `customers scored (random sample of ${s.customers_total.toLocaleString()})`], [pct(s.avg_prob), "average predicted churn probability"],
    [s.tiers.high.toLocaleString(), `high risk (≥70%) · ${pct(s.tiers.high / s.scored)}`], [s.metrics.roc_auc ?? "–", "model ROC-AUC (held-out test)"]];
  $("kpis").replaceChildren(...kp.map(([b, t]) => { const d = document.createElement("div"); d.className = "kpi"; const x = document.createElement("b"); x.textContent = b; const y = document.createElement("span"); y.textContent = t; d.append(x, y); return d; }));
  const top = s.drivers[0];
  $("model-note").textContent = `Model note: "${top.feature}" alone carries ${pct(top.importance)} of the model's feature importance, so predicted risk mostly tracks newsletter status rather than purchase behaviour (see the two charts below). Retention actions therefore also use each customer's actual activity.`;
  bars($("tiers"), ["high", "medium", "low"].map((t) => ({ label: t, value: s.tiers[t], text: `${s.tiers[t]} (${pct(s.tiers[t] / s.scored)})`, cls: t })));
  const mh = Math.max(...s.histogram, 1);
  $("hist").replaceChildren(...s.histogram.map((n, i) => { const d = document.createElement("div"); d.style.height = `${(n / mh) * 100}%`; d.title = `${i * 10}-${i * 10 + 10}%: ${n}`; const l = document.createElement("span"); l.textContent = i * 10; d.append(l); return d; }));
  bars($("by-news"), s.by_news.map((r) => ({ label: `${r.label} (n=${r.n})`, value: r.avg, text: pct(r.avg) })));
  bars($("drivers"), s.drivers.map((r) => ({ label: r.feature, value: r.importance, text: pct(r.importance) })));
  const t = $("risk-table"); t.replaceChildren();
  const head = t.createTHead().insertRow(); ["Customer", "Churn prob.", "Total spend", "Days since last purchase", "Newsletter"].forEach((h) => (head.insertCell().outerHTML = `<th>${h}</th>`));
  const tb = t.createTBody();
  for (const r of s.high_risk) {
    const row = tb.insertRow(); row.onclick = () => viewCustomer(r.customer_id);
    [r.customer_id.slice(0, 16) + "…", pct(r.prob), r.total_spend, r.recency_days, r.news].forEach((v) => (row.insertCell().textContent = v));
  }
}

async function viewCustomer(id) {
  $("cid").value = id; $("cust-msg").textContent = "Loading…"; $("cust").hidden = true;
  let c;
  try { c = await api(`/api/churn/customer/${encodeURIComponent(id)}`); } catch (e) { $("cust-msg").textContent = `Error: ${e.message}`; return; }
  $("cust-msg").textContent = "";
  renderCustomer(c);
}

function renderCustomer(c) {
  $("cust").hidden = false;
  $("prob").textContent = pct(c.probability);
  const b = $("tier-badge"); b.textContent = `${c.tier} risk`; b.className = `badge ${c.tier}`;
  const g = $("gauge-fill"); g.style.width = pct(c.probability); g.style.background = colour[c.tier];
  const dl = $("info"); dl.replaceChildren();
  for (const [k, v] of Object.entries(c.info)) { const dt = document.createElement("dt"); dt.textContent = k.replaceAll("_", " "); const dd = document.createElement("dd"); dd.textContent = v ?? "–"; dl.append(dt, dd); }
  $("recs").replaceChildren(...c.recommendations.map((r) => { const li = document.createElement("li"); const s = document.createElement("small"); s.textContent = `${r.priority} priority`; const t = document.createElement("b"); t.textContent = r.title; li.append(s, t, document.createTextNode(r.detail)); return li; }));
  fill($("sugg"), c.suggested_products);
  $("cust").scrollIntoView({ behavior: "smooth" });
}

$("cid-go").onclick = () => $("cid").value.trim() && viewCustomer($("cid").value.trim());
$("cid").onkeydown = (e) => e.key === "Enter" && $("cid-go").click();
$("cid").oninput = async () => { const q = $("cid").value.trim(); if (q.length < 3) return; $("cid-list").replaceChildren(...(await api(`/api/churn/customers?q=${encodeURIComponent(q)}&limit=10`)).map((i) => { const o = document.createElement("option"); o.value = i; return o; })); };
$("cid-rand").onclick = async () => {
  const hex = "0123456789abcdef", ids = await api(`/api/churn/customers?q=${hex[Math.floor(Math.random() * 16)]}&limit=50`);
  if (ids.length) viewCustomer(ids[Math.floor(Math.random() * ids.length)]);
};
