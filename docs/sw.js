// Keeps Story AI's own files on the device so it opens with no internet.
// The AI model itself is stored separately by the AI library the first time it downloads.
const CACHE = "story-ai-v2";
const LIB = "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.86/lib/index.js";
const SHELL = ["./", "./index.html", "./manifest.webmanifest", "./llm-worker.js", "./icon-180.png", "./icon-192.png", "./icon-512.png", LIB];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith("story-ai-") && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  const isLib = req.url === LIB;
  const isOwn = url.origin === self.location.origin;
  if (!isLib && !isOwn) return; // model downloads go straight to the AI library's own storage

  if (isLib) {
    // The library never changes at this version: use the saved copy.
    e.respondWith(caches.match(LIB).then((hit) => hit || fetch(req).then((res) => {
      const copy = res.clone();
      caches.open(CACHE).then((c) => c.put(LIB, copy));
      return res;
    })));
    return;
  }

  // App files: show the saved copy straight away, and refresh it in the background when online.
  e.respondWith(
    caches.open(CACHE).then(async (c) => {
      const key = req.mode === "navigate" ? "./index.html" : req;
      const hit = await c.match(key, { ignoreSearch: true });
      const fresh = fetch(req).then((res) => {
        if (res.ok) c.put(key, res.clone());
        return res;
      }).catch(() => hit);
      return hit || fresh;
    })
  );
});
