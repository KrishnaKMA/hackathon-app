(() => {
  const api = window.AppModules.api;
  const ui = window.AppModules.ui;
  const animations = window.AppModules.animations;
  const chat = window.AppModules.chat;

  document.body.dataset.page = "home";
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

  let currentFile = null;
  let heatmapShown = false;
  let lastPrediction = null;

  window.showPage = (pageName, evt) => {
    document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
    const page = document.getElementById(pageName);
    if (page) page.classList.add("active");

    document.querySelectorAll(".nav-link").forEach((link) => link.classList.remove("active"));
    if (evt && evt.target) evt.target.classList.add("active");

    document.body.dataset.page = pageName;
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
  }

  // Chat functionality
  function addChatMessage(text, type) {
    if (!chatMessages) return;
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
      const context = lastPrediction ? {
        predicted_class: lastPrediction.predicted_display,
        confidence: lastPrediction.confidence_pct,
        is_tumor: lastPrediction.is_tumor,
        all_predictions: lastPrediction.all_predictions
      } : null;
      
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

  function initFeatureCarousel() {
    const field = document.querySelector(".features.carousel");
    if (!field) return;

    const cards = Array.from(field.querySelectorAll(".feature-card"));
    if (cards.length < 2) return;

    const dotsContainer = document.getElementById("featureDots");
    let dots = [];

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const smallScreen = window.matchMedia("(max-width: 900px)");
    if (reduceMotion.matches || smallScreen.matches) return;

    const total = cards.length;
    let activeIndex = Math.floor(total / 2);
    let lastWheel = 0;

    const wrapIndex = (idx) => (idx + total) % total;

    let rafId = null;
    const render = () => {
      const half = Math.floor(total / 2);
      cards.forEach((card, idx) => {
        let offset = idx - activeIndex;
        if (offset > half) offset -= total;
        if (offset < -half) offset += total;
        const abs = Math.abs(offset);
        const x = offset * 220;
        const y = abs * 18;
        const scale = 1 - Math.min(abs * 0.1, 0.34);
        const opacity = 1 - abs * 0.22;
        const z = 10 - abs;

        card.style.setProperty("--x", `${x}px`);
        card.style.setProperty("--y", `${y}px`);
        card.style.setProperty("--scale", scale);
        card.style.setProperty("--opacity", Math.max(opacity, 0.55));
        card.style.setProperty("--z", z);
        card.classList.toggle("active", idx === activeIndex);
      });

      dots.forEach((dot, idx) => {
        dot.classList.toggle("active", idx === activeIndex);
      });
    };

    const scheduleRender = () => {
      if (rafId) return;
      rafId = requestAnimationFrame(() => {
        rafId = null;
        render();
      });
    };

    const setActive = (idx) => {
      activeIndex = wrapIndex(idx);
      scheduleRender();
    };

    dots = dotsContainer
      ? cards.map((_, idx) => {
          const dot = document.createElement("button");
          dot.type = "button";
          dot.className = "carousel-dot";
          dot.setAttribute("aria-label", `Show card ${idx + 1}`);
          dot.addEventListener("click", () => setActive(idx));
          dotsContainer.appendChild(dot);
          return dot;
        })
      : [];

    cards.forEach((card, idx) => {
      card.addEventListener("click", () => setActive(idx));
    });

    field.addEventListener(
      "wheel",
      (event) => {
        const goingDown = event.deltaY > 0;
        const now = Date.now();
        if (now - lastWheel < 180) return;
        lastWheel = now;

        event.preventDefault();
        setActive(activeIndex + (goingDown ? 1 : -1));
      },
      { passive: false }
    );

    render();
  }

  animations.initScrollReveal();
  initFeatureCarousel();

  const synapsesFrame = document.querySelector(".synapses-frame iframe");
  if (synapsesFrame) {
    synapsesFrame.setAttribute("tabindex", "-1");
  }

  const brainOrbit = document.getElementById("brainOrbit");
  if (brainOrbit) {
    let targetX = 0;
    let targetY = 0;
    let currentX = 0;
    let currentY = 0;

    const update = () => {
      currentX += (targetX - currentX) * 0.08;
      currentY += (targetY - currentY) * 0.08;
      brainOrbit.style.transform = `rotateX(${currentY}deg) rotateY(${currentX}deg)`;
      requestAnimationFrame(update);
    };

    const onMove = (event) => {
      const rect = brainOrbit.getBoundingClientRect();
      const x = (event.clientX - (rect.left + rect.width / 2)) / rect.width;
      const y = (event.clientY - (rect.top + rect.height / 2)) / rect.height;
      targetX = x * 18;
      targetY = -y * 18;
    };

    window.addEventListener("mousemove", onMove);
    update();
  }
  if (chat && chat.init) {
    chat.init({ api });
  }
})();
