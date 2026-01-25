window.AppModules = window.AppModules || {};

window.AppModules.navigation = (() => {
  function showPage(pageName, evt) {
    document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
    const page = document.getElementById(pageName);
    if (page) page.classList.add("active");

    document.querySelectorAll(".nav-link").forEach((link) => link.classList.remove("active"));
    if (evt && evt.target) evt.target.classList.add("active");

    document.body.dataset.page = pageName;
    window.scrollTo(0, 0);
  }

  function init() {
    document.body.dataset.page = "home";
    window.showPage = showPage;
  }

  return { init, showPage };
})();
