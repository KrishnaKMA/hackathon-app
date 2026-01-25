(() => {
  const api = window.AppModules.api;
  const animations = window.AppModules.animations;
  const chat = window.AppModules.chat;
  const navigation = window.AppModules.navigation;
  const home = window.AppModules.home;
  const detector = window.AppModules.detector;
  const learn = window.AppModules.learn;
  const citations = window.AppModules.citations;

  if (navigation && navigation.init) navigation.init();
  if (animations && animations.initScrollReveal) animations.initScrollReveal();
  if (home && home.init) home.init();
  if (detector && detector.init) detector.init();
  if (learn && learn.init) learn.init();
  if (citations && citations.init) citations.init();

  if (chat && chat.init) {
    chat.init({ api });
  }
})();
