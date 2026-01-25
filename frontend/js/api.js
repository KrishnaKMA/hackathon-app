window.AppModules = window.AppModules || {};

window.AppModules.api = (() => {
  const isLocalhost =
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1" ||
    window.location.hostname === "" ||
    window.location.protocol === "file:";

  const API_BASE = isLocalhost ? "http://127.0.0.1:8000" : "/api";

  async function postFile(path, file) {
    const form = new FormData();
    form.append("file", file);

    const res = await fetch(`${API_BASE}${path}`, { method: "POST", body: form });
    if (!res.ok) {
      const text = await res.text();
      throw new Error("Server error (" + res.status + "): " + text);
    }
    return res.json();
  }

  async function postJson(path, data) {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error("Server error (" + res.status + "): " + text);
    }
    return res.json();
  }

  async function predict(file) {
    return postFile("/predict", file);
  }

  async function heatmap(file) {
    return postFile("/heatmap", file);
  }

  async function chat(message, context = null) {
    return postJson("/chat", { message, context });
  }

  return { predict, heatmap, chat };
})();
