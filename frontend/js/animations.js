window.AppModules = window.AppModules || {};

window.AppModules.animations = (() => {
  function initScrollReveal() {
    if (!("IntersectionObserver" in window)) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("in-view");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15 }
    );

    document.querySelectorAll(".reveal").forEach((el) => observer.observe(el));
  }

  return { initScrollReveal };
})();
