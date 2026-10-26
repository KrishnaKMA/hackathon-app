// Single deployment setting. Preserve an explicit override supplied by the host.
window.NEURODETECT_API_BASE_URL = window.NEURODETECT_API_BASE_URL ||
  (["localhost", "127.0.0.1", ""].includes(window.location.hostname)
    ? "http://127.0.0.1:8000"
    : "https://hackathon-app-adb9.onrender.com");
