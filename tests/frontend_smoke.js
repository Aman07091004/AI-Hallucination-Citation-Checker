// Frontend smoke test: runs frontend/index.html in a simulated DOM (jsdom) with a stubbed API.
// Covers submit -> progressive results -> highlighting -> sorting -> selection, and proves that
// untrusted text (the document AND model explanations) is never rendered as markup.
//
//   npm install jsdom      (once, from the repo root)
//   node tests/frontend_smoke.js
const fs = require("fs");
const { JSDOM } = require("jsdom");
const path = require("path");
const html = fs.readFileSync(path.join(__dirname, "..", "frontend", "index.html"), "utf8");

const TEXT = "See Hadley v Baxendale (1854) 9 Ex 341 and <img src=x onerror=window.__pwned=1> then Fake v Case [2020] EWHC 1 (Comm), plus section 37 of the Senior Courts Act 1981.";
const span = (s) => { const i = TEXT.indexOf(s); return [[i, i + s.length]]; };
const XSS = "<img src=x onerror=window.__pwned=1>";

const stubs = [
  { id: 0, kind: "case", name: "Hadley v Baxendale", citation: "(1854) 9 Ex 341", spans: span("Hadley v Baxendale (1854) 9 Ex 341"), status: "pending" },
  { id: 1, kind: "case", name: "Fake v Case", citation: "[2020] EWHC 1 (Comm)", spans: span("Fake v Case [2020] EWHC 1 (Comm)"), status: "pending" },
  { id: 2, kind: "legislation", name: "Senior Courts Act 1981", citation: "Senior Courts Act 1981", spans: span("section 37 of the Senior Courts Act 1981"), status: "pending" },
];
const done0 = { ...stubs[0], status: "done", verdict: "SUPPORTED", category: "verified", depth: "full", confidence_pct: 91, explanation: "Matches. " + XSS, evidence: { excerpt: "the rule in Hadley " + XSS, source: "Local case-law dataset" }, scope_note: "" };
const done1 = { ...stubs[1], status: "done", verdict: "LIKELY_FABRICATED", category: "fabricated", depth: "existence_only", confidence_pct: 88, explanation: "No such judgment found.", evidence: { source: "javascript:alert(1)" }, scope_note: "" };
const done2 = { ...stubs[2], status: "done", verdict: "REAL_BUT_NOT_IN_LOCAL_CORPUS", category: "verified", depth: "existence_only", confidence_pct: 95, explanation: "Real Act.", evidence: { source: "https://www.legislation.gov.uk/ukpga/1981/54" }, scope_note: "Existence checked; text not compared." };

const base = { id: "j1", text: TEXT, warnings: ["Scanned document: read with OCR."], error: null, total: 3 };
const snapshots = [
  { ...base, status: "running", stage: "Checking authorities", done: 0, citations: stubs, summary: { verified: 0, misapplied: 0, fabricated: 0, unresolved: 0, pending: 3 } },
  { ...base, status: "running", stage: "Checking authorities", done: 1, citations: [done0, stubs[1], stubs[2]], summary: { verified: 1, misapplied: 0, fabricated: 0, unresolved: 0, pending: 2 } },
  { ...base, status: "done", stage: "Finished", done: 3, citations: [done0, done1, done2], summary: { verified: 2, misapplied: 0, fabricated: 1, unresolved: 0, pending: 0 } },
];

let pollCount = 0;
const dom = new JSDOM(html, {
  runScripts: "dangerously", pretendToBeVisual: true, url: "http://localhost/",
  beforeParse(window) {
    window.POLL_MS = 5;
    window.Element.prototype.scrollIntoView = function () {};
    window.fetch = async (url, opts = {}) => {
      const json = (body, status = 200) => ({ ok: status < 400, status, json: async () => body });
      if (url.endsWith("/api/health")) return json({ status: "ok", corpus_documents: 58, verification_configured: true, limits: { max_text_chars: 60000, max_upload_mb: 10, max_citations: 30 } });
      if (url.endsWith("/api/verify/text")) { window.__posted = JSON.parse(opts.body); return json({ job_id: "j1" }, 202); }
      if (url.endsWith("/api/jobs/j1")) return json(snapshots[Math.min(pollCount++, snapshots.length - 1)]);
      return json({ detail: "unexpected " + url }, 500);
    };
  },
});
const { window } = dom;
const { document } = window;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failed = 0;
const check = (name, cond, extra = "") => { console.log((cond ? "PASS " : "FAIL ") + name + (cond ? "" : "  " + extra)); if (!cond) failed++; };

(async () => {
  await sleep(50);
  check("footer shows corpus size from /api/health", document.querySelector("#footer-corpus").textContent.includes("58"));

  // empty submit -> friendly error, no request
  document.querySelector("#check-btn").click(); await sleep(10);
  check("empty submit shows an error", !document.querySelector("#error-banner").hidden);

  document.querySelector("#text-input").value = TEXT;
  document.querySelector("#check-btn").click();
  await sleep(300);

  check("text was posted to the API", window.__posted && window.__posted.text === TEXT);
  check("results section is visible", !document.querySelector("#results").hidden);
  check("finished: button re-enabled", document.querySelector("#check-btn").disabled === false);

  const marks = [...document.querySelectorAll("#doc mark.cite")];
  check("3 highlights rendered", marks.length === 3, String(marks.length));
  check("highlight text is exact", marks[0].textContent === "Hadley v Baxendale (1854) 9 Ex 341", marks[0].textContent);
  check("fabricated highlight has fabricated class", marks[1].classList.contains("c-fabricated"));
  check("existence-only verified is marked shallow", marks[2].classList.contains("shallow") && marks[2].classList.contains("c-verified"));
  check("document text is intact around highlights", document.querySelector("#doc").textContent === TEXT);

  check("SECURITY: injected <img> in the document did not become an element", document.querySelectorAll("#doc img, #cards img").length === 0);
  check("SECURITY: onerror handler never ran", window.__pwned === undefined);
  check("injected markup is shown as literal text", document.querySelector("#cards").textContent.includes(XSS));
  check("SECURITY: javascript: source is not rendered as a link", ![...document.querySelectorAll("#cards a")].some((a) => a.href.startsWith("javascript")));
  check("https source IS rendered as a safe link", [...document.querySelectorAll("#cards a")].some((a) => a.href.startsWith("https://www.legislation.gov.uk") && a.rel.includes("noopener")));

  const cards = [...document.querySelectorAll("#cards .card")];
  check("3 cards rendered", cards.length === 3);
  check("flagged-first: fabricated card is first", cards[0].classList.contains("c-fabricated"), cards[0].className);
  check("confidence percentage shown", cards[0].textContent.includes("88%"));
  check("existence-only chip label", cards.some((c) => c.querySelector(".chip").textContent === "Verified (existence only)"));

  const tally = [...document.querySelectorAll("#tally .tally-num")].map((n) => n.textContent);
  check("summary tally is verified 2 / misapplied 0 / fabricated 1 / unresolved 0", tally.join(",") === "2,0,1,0", tally.join(","));
  check("warning banner shown", document.querySelector("#warning-list").textContent.includes("OCR"));
  check("progress bar hidden when finished", document.querySelector("#progress").hidden);
  check("download button available when done", !document.querySelector("#download-btn").hidden);

  marks[1].click(); await sleep(5);
  check("clicking a highlight selects its card", document.querySelector(".card.selected")?.classList.contains("c-fabricated"));
  check("...and marks the highlight selected", document.querySelector("mark.selected") === marks[1]);

  document.querySelector("#sort-order").click(); await sleep(5);
  const ordered = [...document.querySelectorAll("#cards .card .card-name")].map((n) => n.textContent);
  check("document-order sort restores original order", ordered[0] === "Hadley v Baxendale", ordered.join("|"));

  console.log(failed ? `\n${failed} FAILED` : "\nAll frontend checks passed");
  process.exit(failed ? 1 : 0);
})();
