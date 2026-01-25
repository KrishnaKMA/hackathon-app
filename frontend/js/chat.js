window.AppModules = window.AppModules || {};

window.AppModules.chat = (() => {
  function appendMessage(container, role, text) {
    const message = document.createElement("div");
    message.className = `assistant-message ${role}`;
    message.textContent = text;

    container.appendChild(message);
    container.scrollTop = container.scrollHeight;
  }

  function init({ api }) {
    const form = document.getElementById("assistantForm");
    const input = document.getElementById("assistantInput");
    const sendBtn = document.getElementById("assistantSend");
    const messages = document.getElementById("assistantMessages");
    const status = document.getElementById("assistantStatus");
    const toggle = document.getElementById("assistantToggle");
    const panel = document.getElementById("assistantPanel");
    const closeBtn = document.getElementById("assistantClose");

    if (!form || !input || !sendBtn || !messages || !toggle || !panel) return;

    const setOpen = (isOpen) => {
      panel.classList.toggle("open", isOpen);
      panel.setAttribute("aria-hidden", isOpen ? "false" : "true");
      toggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
      if (isOpen) {
        input.focus();
      }
    };

    toggle.addEventListener("click", () => {
      const isOpen = panel.classList.contains("open");
      setOpen(!isOpen);
    });

    if (closeBtn) {
      closeBtn.addEventListener("click", () => setOpen(false));
    }

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") setOpen(false);
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = input.value.trim();
      if (!text) return;

      appendMessage(messages, "user", text);
      input.value = "";
      sendBtn.disabled = true;
      if (status) status.textContent = "Thinking...";

      try {
        const data = await api.chat(text);
        appendMessage(messages, "assistant", data.response || data.reply || "Sorry, I couldn't generate a response.");
        if (status) status.textContent = "Powered by Gemini (server-side).";
      } catch (err) {
        appendMessage(messages, "assistant", "Sorry, I couldn't reach the assistant right now.");
        if (status) status.textContent = "Assistant unavailable. Please try again later.";
        console.error(err);
      } finally {
        sendBtn.disabled = false;
      }
    });
  }

  return { init };
})();
