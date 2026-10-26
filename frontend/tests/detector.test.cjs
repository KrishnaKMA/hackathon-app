const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { JSDOM } = require("jsdom");
const root = path.join(__dirname, "..");
const tick = () => new Promise(resolve => setImmediate(resolve));
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };
const prediction = (name) => ({ predicted_class: name, predicted_display: name, confidence: .8,
  confidence_pct: 80, is_tumor: true, message: `Model predicts ${name}`,
  all_predictions: { [name]: { probability: .8, confidence_pct: 80, display: name, type: "tumor" } } });
function setup(api) {
  const dom = new JSDOM(fs.readFileSync(path.join(root, "index.html"), "utf8"), { url: "http://127.0.0.1:5500", runScripts: "outside-only" });
  const { window } = dom;
  window.URL.createObjectURL = () => "blob:test";
  window.URL.revokeObjectURL = () => {};
  window.eval(fs.readFileSync(path.join(root, "js/ui.js"), "utf8"));
  window.AppModules.api = { ready: async () => ({}), ...api };
  window.eval(fs.readFileSync(path.join(root, "js/pages/detector.js"), "utf8"));
  window.AppModules.detector.init();
  const get = id => window.document.getElementById(id);
  function upload(name) {
    Object.defineProperty(get("file"), "files", { configurable: true, value: [new window.File(["MRI"], name)] });
    get("file").dispatchEvent(new window.Event("change"));
  }
  return { dom, window, get, upload };
}

test("classification renders before pending optional stages and double clicks do not duplicate", async () => {
  const heat = deferred(), ai = deferred(); let count = 0, chatCount = 0;
  const s = setup({ predict: async () => { count++; return prediction("glioma"); }, heatmap: () => heat.promise, chat: () => { chatCount++; return ai.promise; } });
  s.upload("one.webp"); s.get("predictBtn").click(); s.get("predictBtn").click(); await tick();
  assert.equal(count, 1);
  assert.equal(JSON.parse(s.get("rawOutput").textContent).predicted_class, "glioma");
  assert.match(s.get("result").textContent, /glioma/);
  assert.match(s.window.document.querySelector(".heatmap-status").textContent, /Generating Grad-CAM/);
  heat.resolve({ heatmap: "data:image/png;base64,test" }); ai.resolve({ response: "AI offline", status: "degraded" }); await tick();
  assert.match(s.get("result").textContent, /glioma/);
  assert.equal(s.get("heatmap").src, "data:image/png;base64,test");
  assert.equal(chatCount, 0);
  assert.equal(s.window.document.querySelector(".result-stages"), null);
  assert.equal(s.window.document.querySelector(".heatmap-status").parentElement, s.get("preview").closest(".detector-card"));
  s.dom.window.close();
});

test("a stale classification cannot overwrite a newer upload", async () => {
  const old = deferred(); let count = 0;
  const s = setup({ predict: () => ++count === 1 ? old.promise : Promise.resolve(prediction("pituitary")),
    heatmap: async () => ({ heatmap: "data:image/png;base64,new" }), chat: async () => ({ response: "Explanation", status: "ok" }) });
  s.upload("old.webp"); s.get("predictBtn").click(); await tick();
  s.upload("new.webp"); s.get("predictBtn").click(); await tick();
  old.resolve(prediction("glioma")); await tick();
  assert.match(s.get("result").textContent, /pituitary/);
  assert.doesNotMatch(s.get("result").textContent, /glioma/);
  s.dom.window.close();
});

test("stale optional responses and errors preserve current results", async () => {
  const oldHeat = deferred(), oldAI = deferred(); let heatCount = 0, aiCount = 0;
  const s = setup({ predict: async () => prediction("meningioma"),
    heatmap: () => ++heatCount === 1 ? oldHeat.promise : Promise.reject(new Error("offline")),
    chat: () => ++aiCount === 1 ? oldAI.promise : Promise.reject(new Error("offline")) });
  s.upload("old.webp"); s.get("predictBtn").click(); await tick();
  s.upload("new.webp"); s.get("predictBtn").click(); await tick();
  oldHeat.resolve({ heatmap: "data:image/png;base64,old" }); oldAI.resolve({ response: "OLD EXPLANATION", status: "ok" }); await tick();
  assert.match(s.get("result").textContent, /meningioma/);
  assert.doesNotMatch(s.window.document.querySelector(".heatmap-status").textContent, /OLD EXPLANATION/);
  assert.notEqual(s.get("heatmap").src, "data:image/png;base64,old");
  assert.equal(s.window.document.querySelectorAll(".heatmap-status button")[0].hidden, false);
  s.dom.window.close();
});


test("No Tumor keeps original scan, hides heatmap, and never auto-requests chat", async () => {
  let heatCount = 0, chatCount = 0;
  const s = setup({ predict: async () => ({ ...prediction("notumor"), is_tumor: false }),
    heatmap: async () => { heatCount++; return { heatmap: "data:image/png;base64,bad" }; },
    chat: async () => { chatCount++; return { response: "Hello", status: "ok" }; } });
  s.upload("normal.webp"); s.get("predictBtn").click(); await tick();
  assert.equal(heatCount, 0);
  assert.equal(chatCount, 0);
  assert.equal(s.get("heatmapBtn").classList.contains("hidden"), true);
  assert.equal(s.get("heatmap").classList.contains("hidden"), true);
  assert.equal(s.get("preview").classList.contains("hidden"), false);
  assert.equal(s.get("status").textContent, "Analysis complete");
  s.get("detectorChatInput").value = "Hello";
  s.get("detectorChatSendBtn").click(); await tick();
  assert.equal(chatCount, 1);
  assert.match(s.get("detectorChatMessages").textContent, /Hello/);
  s.dom.window.close();
});


test("chat remembers successful turns and clears history for a new scan", async () => {
  const requests = [];
  const s = setup({ chat: async (message, context, signal, history) => {
    requests.push({ message, history }); return { response: "Brief answer.", status: "ok" };
  } });
  async function ask(text) { s.get("detectorChatInput").value = text; s.get("detectorChatSendBtn").click(); await tick(); }
  await ask("What is glioma?");
  await ask("And meningioma?");
  assert.equal(requests[0].history.length, 0);
  assert.deepEqual(requests[1].history.map(x => x.role).join(","), "user,assistant");
  assert.equal(requests[1].history[0].content, "What is glioma?");
  s.upload("different.webp");
  await ask("Hello");
  assert.equal(requests[2].history.length, 0);
  s.dom.window.close();
});
