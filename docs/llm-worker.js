// Runs the on-device AI off the main thread so the page stays smooth while it writes.
import { WebWorkerMLCEngineHandler } from "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.86/lib/index.js";

const handler = new WebWorkerMLCEngineHandler();

// The model reads a prompt in pieces of up to 1-2 thousand words at a time, each one a single job for the
// graphics chip. On a slow or power-saving chip one big piece can take over 2 seconds, and Windows then resets
// the graphics driver (DXGI_ERROR_DEVICE_HUNG); after that the browser blocks the AI until it is restarted.
// Smaller pieces keep every job short. The library takes the size from the model file and offers no setting,
// so it is lowered here before each request. Its memory was set up for the larger size, so smaller is safe.
const PREFILL_CHUNK = 256;
function shrinkPrefill() {
  try {
    const pipes = handler.engine && handler.engine.loadedModelIdToPipeline;
    if (!pipes) return;
    for (const p of pipes.values()) if (p && p.prefillChunkSize > PREFILL_CHUNK) p.prefillChunkSize = PREFILL_CHUNK;
  } catch (e) { /* library internals changed: leave it as it is */ }
}

self.onmessage = (msg) => { shrinkPrefill(); handler.onmessage(msg); };
