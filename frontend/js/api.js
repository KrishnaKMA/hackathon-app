window.AppModules = window.AppModules || {};

window.AppModules.api = (() => {
  const API_BASE = (window.NEURODETECT_API_BASE_URL ||
    (["localhost", "127.0.0.1", ""].includes(window.location.hostname)
      ? "http://127.0.0.1:8000" : window.location.origin)).replace(/\/$/, "");
  const timings = [];

  async function request(path, options = {}) {
    const started = performance.now();
    const res = await fetch(`${API_BASE}${path}`, options);
    timings.push({ path, durationMs: performance.now() - started, status: res.status });
    if (timings.length > 100) timings.shift();
    if (!res.ok) {
      let detail = "Request failed";
      try { detail = (await res.json()).detail || detail; } catch (_) { /* no JSON */ }
      throw new Error(res.status === 503 ? "Model is still loading or unavailable. Retry shortly." : `${res.status}: ${typeof detail === "string" ? detail : "Invalid request"}`);
    }
    return res.json();
  }

  async function postFile(path, file, signal) {
    const form = new FormData();
    form.append("file", file);
    return request(path, { method: "POST", body: form, signal });
  }

  function predict(file, signal) { return postFile("/api/predict", file, signal); }
  function heatmap(file, signal) { return postFile("/api/heatmap", file, signal); }
  function chat(message, context = null, signal, history = []) {
    return request("/api/chat", { method: "POST", signal,
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message, context, history }) });
  }
  function ready(signal) { return request("/api/ready", { signal }); }
  return { predict, heatmap, chat, ready, timings };
})();
