const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { JSDOM } = require("jsdom");
const root = path.join(__dirname, "..");
for (const [page, expected] of [
  ["http://127.0.0.1:5500", "http://127.0.0.1:8000/api/ready"],
  ["http://localhost:5500", "http://127.0.0.1:8000/api/ready"],
  ["https://frontend-4s5x.onrender.com", "https://hackathon-app-adb9.onrender.com/api/ready"]
]) {
  test(`API routing for ${page}`, async () => {
    const dom = new JSDOM("", { url: page, runScripts: "outside-only" });
    let requested;
    dom.window.fetch = async (url) => { requested = url; return { ok: true, status: 200, json: async () => ({ status: "ready" }) }; };
    dom.window.eval(fs.readFileSync(path.join(root, "js/config.js"), "utf8"));
    dom.window.eval(fs.readFileSync(path.join(root, "js/api.js"), "utf8"));
    await dom.window.AppModules.api.ready();
    assert.equal(requested, expected);
    dom.window.close();
  });
}
test("host API override survives deployment defaults", async () => {
  const dom = new JSDOM("", {url:"https://custom.example",runScripts:"outside-only"});
  dom.window.NEURODETECT_API_BASE_URL = "https://custom-api.example/";
  let requested;
  dom.window.fetch = async (url) => { requested = url; return {ok:true,status:200,json:async () => ({})}; };
  dom.window.eval(fs.readFileSync(path.join(root,"js/config.js"),"utf8"));
  dom.window.eval(fs.readFileSync(path.join(root,"js/api.js"),"utf8"));
  await dom.window.AppModules.api.ready();
  assert.equal(requested,"https://custom-api.example/api/ready");
  dom.window.close();
});
