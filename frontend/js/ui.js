window.AppModules = window.AppModules || {};

window.AppModules.ui = (() => {
  function toggleHidden(el, hidden) {
    if (!el) return;
    el.classList.toggle("hidden", Boolean(hidden));
  }

  function setStatus(statusEl, text, type) {
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.className = "status-badge";
    if (type === "success") statusEl.classList.add("success");
    else if (type === "error") statusEl.classList.add("error");
  }

  function renderResult(resultEl, data) {
    if (!resultEl) return;
    const classes = data.all_predictions || {};
    const isTumor = data.is_tumor !== false;
    const predClass = data.predicted_display || "Unknown";

    const sorted = Object.entries(classes).sort(([, a], [, b]) => b.probability - a.probability);

    const predictionCards = sorted
      .map(([, info], idx) => {
        const isSelected = idx === 0;
        const pct = info.confidence_pct || 0;
        return `
          <div class="prediction-card ${isSelected ? "selected" : ""}">
            <div class="class-name">${info.display}</div>
            <div class="class-bar">
              <div class="class-fill" style="width: ${pct}%;"></div>
            </div>
            <div class="class-prob">${pct.toFixed(1)}%</div>
            <div class="class-type">${info.type}</div>
          </div>
        `;
      })
      .join("");

    resultEl.innerHTML = `
      <div class="main-result ${isTumor ? "tumor" : "normal"}">
        <div class="result-title">${predClass}</div>
        <div class="result-message">
          ${data.message || "Analysis complete"}
        </div>
      </div>
      <div class="result-grid">
        ${predictionCards}
      </div>
    `;

    const rawOutput = document.getElementById("rawOutput");
    if (rawOutput) {
      rawOutput.textContent = JSON.stringify(data, null, 2);
    }
  }

  function renderLoading(resultEl) {
    if (!resultEl) return;
    resultEl.innerHTML = `
      <div class="loading-container">
        <div class="loading-spinner"></div>
        <div class="loading-text">Running prediction...</div>
      </div>
    `;
    const rawOutput = document.getElementById("rawOutput");
    if (rawOutput) rawOutput.textContent = "{ }";
  }

  function showError(resultEl, message) {
    if (!resultEl) return;
    resultEl.innerHTML = `
      <div class="error-card">
        <div class="error-title">Request failed</div>
        <div class="error-message">${message}</div>
      </div>
    `;
    const rawOutput = document.getElementById("rawOutput");
    if (rawOutput) rawOutput.textContent = "{ }";
  }

  function setPreviewImage(previewImg, url) {
    if (!previewImg) return;
    previewImg.src = url;
  }

  function showPreview(previewImg, heatmapImg) {
    toggleHidden(previewImg, false);
    toggleHidden(heatmapImg, true);
  }

  function showHeatmap(previewImg, heatmapImg, src) {
    if (heatmapImg && src) heatmapImg.src = src;
    toggleHidden(heatmapImg, false);
    toggleHidden(previewImg, true);
  }

  function setHeatmapButton(heatmapBtn, { text, disabled, visible }) {
    if (!heatmapBtn) return;
    if (typeof text === "string") heatmapBtn.textContent = text;
    heatmapBtn.disabled = Boolean(disabled);
    if (typeof visible === "boolean") toggleHidden(heatmapBtn, !visible);
  }

  return {
    setStatus,
    renderResult,
    renderLoading,
    showError,
    setPreviewImage,
    showPreview,
    showHeatmap,
    setHeatmapButton,
    toggleHidden,
  };
})();
