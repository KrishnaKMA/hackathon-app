(() => {
  const api = window.AppModules.api;
  const ui = window.AppModules.ui;
  const animations = window.AppModules.animations;

  const fileInput = document.getElementById("file");
  const previewImg = document.getElementById("preview");
  const heatmapImg = document.getElementById("heatmap");
  const predictBtn = document.getElementById("predictBtn");
  const heatmapBtn = document.getElementById("heatmapBtn");
  const statusEl = document.getElementById("status");
  const resultEl = document.getElementById("result");

  let currentFile = null;
  let heatmapShown = false;

  window.showPage = (pageName, evt) => {
    document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
    const page = document.getElementById(pageName);
    if (page) page.classList.add("active");

    document.querySelectorAll(".nav-link").forEach((link) => link.classList.remove("active"));
    if (evt && evt.target) evt.target.classList.add("active");

    window.scrollTo(0, 0);
  };

  function resetPreview() {
    heatmapShown = false;
    ui.showPreview(previewImg, heatmapImg);
    ui.setHeatmapButton(heatmapBtn, { visible: false, text: "Show Heatmap", disabled: false });
  }

  if (fileInput) {
    fileInput.addEventListener("change", () => {
      const file = fileInput.files?.[0];
      if (!file) return;

      currentFile = file;
      resetPreview();

      const url = URL.createObjectURL(file);
      ui.setPreviewImage(previewImg, url);
      ui.setStatus(statusEl, "Ready", "success");
      if (resultEl) resultEl.innerHTML = "";
    });
  }

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

  if (predictBtn) {
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
  }

  animations.initScrollReveal();
})();
