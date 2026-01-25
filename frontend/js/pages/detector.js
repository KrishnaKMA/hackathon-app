window.AppModules = window.AppModules || {};

window.AppModules.detector = (() => {
  function init() {
    const api = window.AppModules.api;
    const ui = window.AppModules.ui;

    const fileInput = document.getElementById("file");
    const previewImg = document.getElementById("preview");
    const heatmapImg = document.getElementById("heatmap");
    const predictBtn = document.getElementById("predictBtn");
    const heatmapBtn = document.getElementById("heatmapBtn");
    const statusEl = document.getElementById("status");
    const resultEl = document.getElementById("result");
    const chatInput = document.getElementById("detectorChatInput");
    const chatSendBtn = document.getElementById("detectorChatSendBtn");
    const chatMessages = document.getElementById("detectorChatMessages");

    if (!fileInput || !predictBtn || !previewImg || !heatmapImg || !statusEl || !resultEl) {
      return;
    }

    let currentFile = null;
    let heatmapShown = false;
    let lastPrediction = null;

    function resetPreview() {
      heatmapShown = false;
      ui.showPreview(previewImg, heatmapImg);
      ui.setHeatmapButton(heatmapBtn, { visible: false, text: "Show Heatmap", disabled: false });
    }

    fileInput.addEventListener("change", () => {
      const file = fileInput.files?.[0];
      if (!file) return;

      currentFile = file;
      resetPreview();

      const url = URL.createObjectURL(file);
      ui.setPreviewImage(previewImg, url);
      ui.setStatus(statusEl, "Ready", "success");
      resultEl.textContent = "Upload an MRI scan to generate results.";
    });

    if (heatmapBtn) {
      heatmapBtn.addEventListener("click", async () => {
        if (!currentFile) return;

        heatmapShown = !heatmapShown;
        if (heatmapShown) {
          try {
            ui.setHeatmapButton(heatmapBtn, { text: "Loading heatmap...", disabled: true, visible: true });
            const data = await api.heatmap(currentFile);
            ui.showHeatmap(previewImg, heatmapImg, data.heatmap);
            ui.setHeatmapButton(heatmapBtn, { text: "Hide Heatmap", disabled: false, visible: true });
          } catch (err) {
            console.error(err);
            alert("Failed to generate heatmap: " + (err?.message ?? err));
            heatmapShown = false;
            ui.setHeatmapButton(heatmapBtn, { text: "Show Heatmap", disabled: false, visible: true });
            ui.showPreview(previewImg, heatmapImg);
          }
        } else {
          ui.showPreview(previewImg, heatmapImg);
          ui.setHeatmapButton(heatmapBtn, { text: "Show Heatmap", disabled: false, visible: true });
        }
      });
    }

    predictBtn.addEventListener("click", async () => {
      const file = fileInput?.files?.[0];
      if (!file) {
        ui.setStatus(statusEl, "Pick an image first", "error");
        return;
      }

      try {
        ui.setStatus(statusEl, "Analyzing...", "info");
        ui.renderLoading(resultEl);

        const data = await api.predict(file);
        lastPrediction = data;
        ui.renderResult(resultEl, data);
        ui.setHeatmapButton(heatmapBtn, { visible: true, text: "Show Heatmap", disabled: false });
        heatmapShown = false;
        ui.setStatus(statusEl, "Complete", "success");
      } catch (err) {
        console.error(err);
        ui.setStatus(statusEl, "Error", "error");
        ui.setHeatmapButton(heatmapBtn, { visible: false, text: "Show Heatmap", disabled: false });
        ui.showError(resultEl, String(err?.message ?? err));
      }
    });

    function addChatMessage(text, type) {
      if (!chatMessages) return null;
      const msgEl = document.createElement("div");
      msgEl.className = `chat-message ${type}`;
      msgEl.textContent = text;
      chatMessages.appendChild(msgEl);
      chatMessages.scrollTop = chatMessages.scrollHeight;
      return msgEl;
    }

    async function sendChatMessage() {
      if (!chatInput || !chatInput.value.trim()) return;

      const message = chatInput.value.trim();
      chatInput.value = "";

      addChatMessage(message, "user");
      const loadingMsg = addChatMessage("Thinking...", "loading");

      try {
        const context = lastPrediction
          ? {
              predicted_class: lastPrediction.predicted_display,
              confidence: lastPrediction.confidence_pct,
              is_tumor: lastPrediction.is_tumor,
              all_predictions: lastPrediction.all_predictions,
            }
          : null;

        const data = await api.chat(message, context);

        if (loadingMsg) loadingMsg.remove();

        if (data.error) {
          addChatMessage(data.error, "error");
        } else {
          addChatMessage(data.response, "assistant");
        }
      } catch (err) {
        if (loadingMsg) loadingMsg.remove();
        addChatMessage(`Error: ${err?.message ?? err}`, "error");
      }
    }

    if (chatSendBtn) {
      chatSendBtn.addEventListener("click", sendChatMessage);
    }

    if (chatInput) {
      chatInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
          sendChatMessage();
        }
      });
    }
  }

  return { init };
})();
