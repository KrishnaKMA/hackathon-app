window.AppModules = window.AppModules || {};

window.AppModules.detector = (() => {
  function init() {
    const { api, ui } = window.AppModules;
    const get = (id) => document.getElementById(id);
    const fileInput = get("file"), predictBtn = get("predictBtn");
    const preview = get("preview"), heatmapImg = get("heatmap"), heatmapBtn = get("heatmapBtn");
    const status = get("status"), result = get("result");
    const legend = get("heatmapLegend");
    const chatInput = get("detectorChatInput"), sendBtn = get("detectorChatSendBtn"), messages = get("detectorChatMessages");
    if (!fileInput || !predictBtn || !preview || !heatmapImg || !result) return;

    const stages = document.createElement("div");
    stages.className = "heatmap-status";
    stages.setAttribute("aria-live", "polite");
    (legend || preview).after(stages);
    const heatStatus = document.createElement("span");
    const retryHeat = document.createElement("button");
    retryHeat.textContent = "Retry heatmap";
    stages.append(heatStatus, retryHeat);
    retryHeat.hidden = true;
    let version = 0, controller = new AbortController(), lastPrediction = null;
    let heatmapData = null, showingHeatmap = false, previewUrl = null, busy = false;
    const current = (id) => id === version;
    function reset() {
      version += 1; controller.abort(); controller = new AbortController();
      lastPrediction = null; heatmapData = null; showingHeatmap = false; busy = false;
      if (legend) legend.hidden = true;
      predictBtn.disabled = false;
      if (sendBtn) sendBtn.disabled = false;
      if (messages) messages.replaceChildren();
      heatStatus.textContent = "";
      retryHeat.hidden = true;
      ui.showPreview(preview, heatmapImg);
      ui.setHeatmapButton(heatmapBtn, { visible: false, disabled: false, text: "Show heatmap" });
    }
    fileInput.addEventListener("change", () => {
      reset();
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      const file = fileInput.files?.[0];
      if (!file) { preview.removeAttribute("src"); result.textContent = "Choose an MRI image."; return; }
      previewUrl = URL.createObjectURL(file); ui.setPreviewImage(preview, previewUrl);
      result.textContent = "Image selected. Press Predict to analyze.";
      ui.setStatus(status, "Ready to analyze", "success");
    });
    function context() {
      return lastPrediction ? { predicted_class: lastPrediction.predicted_class,
        confidence: lastPrediction.confidence, all_predictions: lastPrediction.all_predictions } : null;
    }
    async function loadHeatmap(id, file, signal) {
      heatStatus.textContent = "Generating Grad-CAM..."; retryHeat.hidden = true;
      try {
        const data = await api.heatmap(file, signal);
        if (!current(id)) return;
        heatmapData = data.heatmap; showingHeatmap = true;
        ui.showHeatmap(preview, heatmapImg, heatmapData);
        heatStatus.textContent = "Grad-CAM ready.";
        if (legend) legend.hidden = false;
        ui.setHeatmapButton(heatmapBtn, { visible: true, disabled: false, text: "Show original" });
      } catch (error) {
        if (!current(id) || error.name === "AbortError") return;
        heatStatus.textContent = "Heatmap unavailable. Prediction is still ready.";
        retryHeat.hidden = false;
      }
    }
    predictBtn.addEventListener("click", async () => {
      const file = fileInput.files?.[0];
      if (!file || busy) return;
      reset(); busy = true; predictBtn.disabled = true;
      const id = version, signal = controller.signal;
      ui.setStatus(status, "Analyzing...", "info"); ui.renderLoading(result);
      try {
        const data = await api.predict(file, signal);
        if (!current(id)) return;
        lastPrediction = data; ui.renderResult(result, data);
        ui.setStatus(status, "Analysis complete", "success");
        // Classification is rendered before independent optional requests start.
        if (data.predicted_class === "notumor" || data.is_tumor === false) {
          heatStatus.textContent = "Original scan shown for No Tumor prediction.";
        } else {
          void loadHeatmap(id, file, signal);
        }
      } catch (error) {
        if (!current(id) || error.name === "AbortError") return;
        ui.setStatus(status, "Prediction unavailable ... retry", "error");
        ui.showError(result, error.message);
      } finally {
        if (current(id)) { busy = false; predictBtn.disabled = false; }
      }
    });
    heatmapBtn?.addEventListener("click", () => {
      if (!heatmapData) return;
      showingHeatmap = !showingHeatmap;
      if (showingHeatmap) ui.showHeatmap(preview, heatmapImg, heatmapData);
      else ui.showPreview(preview, heatmapImg);
      if (legend) legend.hidden = !showingHeatmap;
      heatmapBtn.textContent = showingHeatmap ? "Show original" : "Show Grad-CAM";
    });
    retryHeat.addEventListener("click", () => {
      if (lastPrediction) void loadHeatmap(version, fileInput.files[0], controller.signal);
    });
    function addMessage(text, type) {
      if (!messages) return null;
      const item = document.createElement("div"); item.className = `chat-message ${type}`;
      item.textContent = text; messages.append(item); messages.scrollTop = messages.scrollHeight;
      return item;
    }
    async function sendChat() {
      const text = chatInput?.value.trim();
      if (!text || sendBtn?.disabled) return;
      const id = version; chatInput.value = ""; if (sendBtn) sendBtn.disabled = true;
      addMessage(text, "user"); const loading = addMessage("Thinking...", "loading");
      try {
        const data = await api.chat(text, context(), controller.signal);
        if (!current(id)) return;
        addMessage(data.response, data.status === "ok" ? "assistant" : "error");
      } catch (error) {
        if (current(id) && error.name !== "AbortError") addMessage("Assistant unavailable. Please retry your question.", "error");
      } finally {
        loading?.remove(); if (current(id) && sendBtn) sendBtn.disabled = false;
      }
    }
    sendBtn?.addEventListener("click", sendChat);
    chatInput?.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); void sendChat(); } });
    // One readiness check on opening. No artificial hosting keep-alive polling.
    api.ready().then(() => { if (!fileInput.files?.length && !busy) ui.setStatus(status, "Model ready", "success"); })
      .catch(() => { if (!fileInput.files?.length && !busy) ui.setStatus(status, "Backend warming up ... retry prediction shortly", "info"); });
  }
  return { init };
})();
