window.AppModules = window.AppModules || {};

window.AppModules.home = (() => {
  function init() {
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
  }

  return { init };
})();
