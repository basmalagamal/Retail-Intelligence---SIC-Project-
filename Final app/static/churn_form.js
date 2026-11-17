let formSchema = null;
const label = (n) => n.replaceAll("_", " ");
const GROUPS = ["Profile", "Lifetime activity", "Recent activity", "Baskets", "Preferences"];

async function buildForm() {
  formSchema = (await api("/api/churn/schema")).fields;
  const root = $("cform-fields"); root.replaceChildren();
  for (const g of GROUPS) {
    const fields = formSchema.filter((f) => f.group === g);
    if (!fields.length) continue;
    const fs = document.createElement("fieldset"), lg = document.createElement("legend"), grid = document.createElement("div");
    lg.textContent = g; grid.className = "fgrid"; fs.append(lg, grid);
    for (const f of fields) {
      const w = document.createElement("div"); w.className = "field"; w.dataset.name = f.name;
      const l = document.createElement("label"); l.textContent = label(f.name); l.htmlFor = `f-${f.name}`;
      let inp;
      if (f.kind === "select") {
        inp = document.createElement("select");
        inp.append(new Option("— missing —", ""));
        for (const o of f.options) inp.append(new Option(f.dtype.startsWith("int") ? (o === "1" ? "Yes (1)" : "No (0)") : o, o));
      } else {
        inp = document.createElement("input"); inp.type = "number"; inp.step = "any";
        if (f.min !== null) inp.min = f.min; if (f.max !== null) inp.max = f.max;
        inp.placeholder = f.min !== null && f.max !== null ? `${f.min}–${f.max}` : f.min !== null ? `≥ ${f.min}` : "number";
      }
      inp.id = `f-${f.name}`; inp.name = f.name;
      const e = document.createElement("div"); e.className = "err";
      w.append(l, inp, e); grid.append(w);
    }
    root.append(fs);
  }
}

function setErrors(errs) {
  document.querySelectorAll("#cform .field").forEach((w) => { const m = errs[w.dataset.name]; w.classList.toggle("bad", !!m); w.querySelector(".err").textContent = m || ""; });
}

function collect() {
  const values = {}, errs = {};
  for (const f of formSchema) {
    const el = $(`f-${f.name}`), raw = el.value.trim();
    if (raw === "") { values[f.name] = null; continue; }
    if (f.kind === "number") {
      const x = Number(raw);
      if (!Number.isFinite(x)) errs[f.name] = "must be a number";
      else if (f.min !== null && x < f.min) errs[f.name] = `must be ≥ ${f.min}`;
      else if (f.max !== null && x > f.max) errs[f.name] = `must be ≤ ${f.max}`;
      values[f.name] = x;
    } else values[f.name] = raw;
  }
  return { values, errs };
}

function fillForm(values) {
  for (const f of formSchema) { const v = values[f.name]; $(`f-${f.name}`).value = v === null || v === undefined ? "" : String(v); }
  setErrors({});
}

async function openForm() {
  $("cform-modal").hidden = false;
  if (!formSchema) {
    $("cform-msg").textContent = "Loading fields…";
    try { await buildForm(); $("cform-msg").textContent = ""; } catch (e) { $("cform-msg").textContent = `Error: ${e.message}`; }
  }
}

$("add-cust").onclick = openForm;
$("cform-close").onclick = () => ($("cform-modal").hidden = true);
$("cform-modal").onclick = (e) => { if (e.target.id === "cform-modal") $("cform-modal").hidden = true; };
$("cform-clear").onclick = () => { if (formSchema) { fillForm({}); $("cform-msg").textContent = ""; } };
$("cform-example").onclick = async () => { if (!formSchema) return; try { fillForm((await api("/api/churn/example")).values); $("cform-msg").textContent = "Prefilled from a random customer in the scored sample - edit any value."; } catch (e) { $("cform-msg").textContent = `Error: ${e.message}`; } };

$("cform").onsubmit = async (ev) => {
  ev.preventDefault();
  if (!formSchema) return;
  const { values, errs } = collect();
  setErrors(errs);
  if (Object.keys(errs).length) { $("cform-msg").textContent = "Please fix the highlighted fields."; return; }
  $("cform-msg").textContent = "Predicting…"; $("cform-submit").disabled = true;
  try {
    const r = await fetch("/api/churn/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ values }) });
    const data = await r.json();
    if (!r.ok) {
      const e = (data.detail && data.detail.errors) || {};
      setErrors(e);
      $("cform-msg").textContent = Object.keys(e).length ? "Please fix the highlighted fields." : `Error: ${data.detail || r.statusText}`;
      return;
    }
    $("cform-msg").textContent = "";
    $("cform-modal").hidden = true;
    $("cust-msg").textContent = data.warnings.length ? `Note: ${data.warnings.join(" ")}` : "";
    $("cid").value = "";
    renderCustomer(data);
  } catch (err) {
    $("cform-msg").textContent = `Error: ${err.message}`;
  } finally { $("cform-submit").disabled = false; }
};
