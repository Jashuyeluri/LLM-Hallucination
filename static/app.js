const sourceFileInput = document.getElementById("source-file");
const responseFileInput = document.getElementById("response-file");
const sourceFilename = document.getElementById("source-filename");
const responseFilename = document.getElementById("response-filename");
const sourceText = document.getElementById("source-text");
const sourceUrls = document.getElementById("source-urls"); // Optional: not present in the supplied index.html.
const responseText = document.getElementById("response-text");
const modeSelect = document.getElementById("mode-select");
const policySelect = document.getElementById("policy-select");
const runBtn = document.getElementById("run-btn");
const statusRow = document.getElementById("status-row");
const statusText = document.getElementById("status-text");
const results = document.getElementById("results");
const metricsStrip = document.getElementById("metrics-strip");
const phaseReport = document.getElementById("phase-report");
const abstentionNote = document.getElementById("abstention-note");
const sourceManifestView = document.getElementById("source-manifest");
const evidenceList = document.getElementById("evidence-list");
const sourcePanel = document.getElementById("source-panel");
const sourceHint = document.getElementById("source-hint");
const sourceView = document.getElementById("source-view");
const consistencyPanel = document.getElementById("consistency-panel");
const conflictList = document.getElementById("conflict-list");
const correctionPanel = document.getElementById("correction-panel");
const correctionText = document.getElementById("correction-text");
const iterationTimeline = document.getElementById("iteration-timeline");
const allClear = document.getElementById("all-clear");
const allClearText = document.getElementById("all-clear-text");
const footerText = document.getElementById("footer-text");
const certificateDownload = document.getElementById("download-certificate");

const statusMessages = [
  "Extracting atomic claims…",
  "Cross-referencing against source…",
  "Checking evidence and verifier agreement…",
  "Scoring faithfulness…",
  "Drafting and re-verifying the correction…",
];

const STOP_REASONS = {
  no_errors: "no contradictions found",
  converged: "converged",
  max_iterations: "stopped at the pass limit",
  no_change: "stopped, the rewrite changed nothing",
  correction_disabled: "correction disabled",
};

let statusInterval = null;
let lastData = null;
let lastSubmittedResponse = "";

const el = (tag, className, text) => {
  const node = document.createElement(tag);

  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;

  return node;
};

const labelText = (label) => label.replace(/_/g, " ");

const startStatusCycle = () => {
  let i = 0;

  statusText.textContent = statusMessages[0];

  statusInterval = setInterval(() => {
    i = (i + 1) % statusMessages.length;
    statusText.textContent = statusMessages[i];
  }, 2600);
};

const stopStatusCycle = () => {
  if (statusInterval !== null) {
    clearInterval(statusInterval);
    statusInterval = null;
  }
};

const animateCount = (node, from, to, duration) => {
  const start = performance.now();

  const tick = (now) => {
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);

    node.textContent = Math.round(from + (to - from) * eased);

    if (t < 1) requestAnimationFrame(tick);
  };

  requestAnimationFrame(tick);
};

const mergeSpans = (spans) => {
  const sorted = [...spans].sort((a, b) => a.start - b.start);
  const merged = [];

  sorted.forEach((span) => {
    const last = merged[merged.length - 1];

    if (last && span.start <= last.end) {
      last.end = Math.max(last.end, span.end);
    } else {
      merged.push({
        start: span.start,
        end: span.end,
      });
    }
  });

  return merged;
};

const renderSource = (text, evidence, label) => {
  sourceView.textContent = "";

  const spans = mergeSpans(evidence || []);
  let cursor = 0;
  let firstMark = null;

  spans.forEach((span) => {
    if (span.start > cursor) {
      sourceView.appendChild(
        document.createTextNode(text.slice(cursor, span.start))
      );
    }

    const mark = el(
      "mark",
      `source-mark source-mark--${label}`,
      text.slice(span.start, span.end)
    );

    if (!firstMark) firstMark = mark;

    sourceView.appendChild(mark);
    cursor = span.end;
  });

  if (cursor < text.length) {
    sourceView.appendChild(
      document.createTextNode(text.slice(cursor))
    );
  }

  if (firstMark) {
    firstMark.scrollIntoView({
      block: "nearest",
      behavior: "smooth",
    });
  }
};

const selectClaim = (card, result) => {
  evidenceList
    .querySelectorAll(".evidence-card")
    .forEach((node) => node.classList.remove("is-selected"));

  card.classList.add("is-selected");

  renderSource(
    lastData.source_text,
    result.evidence,
    result.label
  );

  sourceHint.textContent = result.evidence.length
    ? `Showing the evidence behind: “${result.claim}”`
    : "No source sentence was cited for this claim.";
};

const buildVerifierLine = (result) => {
  if (!result.verifier || result.verifier === "llm") {
    return null;
  }

  const parts = [];

  if (result.llm_label) {
    parts.push(`LLM: ${labelText(result.llm_label)}`);
  }

  if (result.nli_label) {
    parts.push(`NLI: ${labelText(result.nli_label)}`);
  }

  if (result.agreement === true) parts.push("agree");
  if (result.agreement === false) parts.push("disagree");

  if (result.llm_label === null) {
    parts.push("resolved by NLI alone");
  }

  return el("span", "evidence-verifiers", parts.join(" · "));
};

const buildCard = (result, index) => {
  const card = el(
    "div",
    `evidence-card evidence-card--${result.label}`
  );

  card.style.animationDelay = `${index * 70}ms`;
  card.tabIndex = 0;
  card.setAttribute("role", "button");

  const main = el("div", "evidence-main");

  main.appendChild(
    el("span", "evidence-claim", result.claim)
  );

  (result.evidence || []).slice(0, 2).forEach((item) => {
    main.appendChild(
      el("blockquote", "evidence-quote", item.text)
    );
  });

  const verifierLine = buildVerifierLine(result);

  if (verifierLine) {
    main.appendChild(verifierLine);
  }

  if (
    result.evidence &&
    result.evidence[0] &&
    result.evidence[0].source_name
  ) {
    main.appendChild(
      el(
        "span",
        "evidence-verifiers",
        `source: ${result.evidence[0].source_name}`
      )
    );
  }

  if (result.reason) {
    main.appendChild(
      el(
        "span",
        "evidence-verifiers",
        `downgraded: ${result.reason.replace(/_/g, " ")}`
      )
    );
  }

  if (result.internally_inconsistent) {
    main.appendChild(
      el(
        "span",
        "evidence-flag",
        "contradicts another claim in this response"
      )
    );
  }

  const meta = el("span", "evidence-meta");

  meta.appendChild(
    el("span", "evidence-score", result.confidence.toFixed(3))
  );

  meta.appendChild(
    el(
      "span",
      `evidence-stamp evidence-stamp--${result.label}`,
      labelText(result.label)
    )
  );

  card.appendChild(main);
  card.appendChild(meta);

  card.addEventListener("click", () => {
    selectClaim(card, result);
  });

  card.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectClaim(card, result);
    }
  });

  return card;
};

const buildChip = (name, value) => {
  const chip = el("span", "metric-chip");

  chip.appendChild(el("span", "metric-name", name));
  chip.appendChild(el("span", "metric-value", value));

  return chip;
};

const renderMetrics = (data) => {
  const m = data.metrics;

  metricsStrip.textContent = "";

  const modeValue =
    m.mode_requested !== m.mode_used
      ? `${m.mode_used} (NLI unavailable)`
      : m.mode_used;

  metricsStrip.appendChild(buildChip("verifier", modeValue));

  metricsStrip.appendChild(
    buildChip(
      "LLM calls",
      `${m.llm_calls_total} total · ${m.llm_calls_made} verifying`
    )
  );

  if (m.mode_used !== "llm") {
    metricsStrip.appendChild(
      buildChip(
        "resolved by NLI",
        `${m.claims_resolved_by_nli}/${m.claims_total}`
      )
    );
  }

  metricsStrip.appendChild(
    buildChip("time", `${(m.latency_ms_total / 1000).toFixed(1)}s`)
  );

  if (m.calibrated) {
    metricsStrip.appendChild(
      buildChip("confidence", "calibrated")
    );
  }

  if (m.rag) {
    metricsStrip.appendChild(
      buildChip(
        "retrieval",
        m.rag.retrieval_mode.replace(/_/g, " ")
      )
    );

    metricsStrip.appendChild(
      buildChip("retrieved chunks", String(m.rag.chunks_retrieved))
    );

    metricsStrip.appendChild(
      buildChip(
        "context reduced",
        `${m.rag.context_reduction_pct}%`
      )
    );
  }

  metricsStrip.appendChild(
    buildChip(
      "correction",
      STOP_REASONS[data.stop_reason] || data.stop_reason
    )
  );
};

const renderConflicts = (data) => {
  conflictList.textContent = "";

  const conflicts = data.consistency_conflicts || [];

  consistencyPanel.hidden = conflicts.length === 0;

  conflicts.forEach((conflict) => {
    const item = el("li", "conflict-item");

    item.appendChild(
      el("span", "conflict-claim", conflict.claim_a)
    );

    item.appendChild(
      el("span", "conflict-vs", "vs")
    );

    item.appendChild(
      el("span", "conflict-claim", conflict.claim_b)
    );

    conflictList.appendChild(item);
  });
};

const renderTimeline = (iterations, fallbackText, score) => {
  iterationTimeline.textContent = "";

  let passes = Array.isArray(iterations) ? [...iterations] : [];

  // If the backend has no correction iterations, show the real
  // original verification result instead of an empty timeline.
  if (passes.length === 0 && fallbackText) {
    passes = [{
      iteration: 0,
      text: fallbackText,
      score: {
        faithfulness_score: score.faithfulness_score,
      },
      contradicted: score.contradicted,
      unsupported: score.unsupported,
      needs_review: score.needs_review,
    }];
  }

  passes.forEach((round, index) => {
    const item = el("li", "timeline-item");

    const title =
      round.iteration === 0 || index === 0
        ? "Original response"
        : `Pass ${round.iteration}`;

    const passScore =
      round.score?.faithfulness_score ??
      score.faithfulness_score;

    const contradicted =
      round.contradicted ?? score.contradicted;

    const unsupported =
      round.unsupported ?? score.unsupported;

    const needsReview =
      round.needs_review ?? score.needs_review;

    const summary =
      `${passScore}% faithful · ` +
      `${contradicted} contradicted · ` +
      `${unsupported} unsupported · ` +
      `${needsReview} needs review`;

    item.appendChild(
      el("span", "timeline-title", title)
    );

    item.appendChild(
      el("span", "timeline-summary", summary)
    );

    const details = el("details", "timeline-details");

    details.appendChild(
      el("summary", "", "Show text")
    );

    details.appendChild(
      el("p", "", round.text || "")
    );

    item.appendChild(details);
    iterationTimeline.appendChild(item);
  });
};

const clearMessage = (score) => {
  if (
    score.contradicted === 0 &&
    score.unsupported === 0 &&
    score.needs_review === 0
  ) {
    return "Every claim checks out. No correction needed.";
  }

  if (score.contradicted === 0) {
    return "No contradictions found. Claims the source cannot confirm were left unchanged.";
  }

  return "Correction was not produced.";
};

const renderResults = (data) => {
  lastData = data;

  certificateDownload.disabled = !data.certificate;

  const {
    score,
    results: claimResults,
    corrected_response: corrected,
  } = data;

  document.getElementById("stat-supported").textContent =
    score.supported;

  document.getElementById("stat-contradicted").textContent =
    score.contradicted;

  document.getElementById("stat-unsupported").textContent =
    score.unsupported;

  document.getElementById("stat-needs-review").textContent =
    score.needs_review;

  document.getElementById("stat-total").textContent =
    score.total_claims;

  const pct = score.faithfulness_score;

  const gaugeFill = document.getElementById("gauge-fill");
  const circumference = 427;

  gaugeFill.style.strokeDashoffset =
    circumference - (pct / 100) * circumference;

  gaugeFill.style.stroke =
    pct >= 70
      ? "#2E9E83"
      : pct >= 40
        ? "#B8791F"
        : "#C1473C";

  animateCount(
    document.getElementById("gauge-number"),
    0,
    Math.round(pct),
    900
  );

  renderMetrics(data);

  evidenceList.textContent = "";

  claimResults.forEach((result, index) => {
    evidenceList.appendChild(buildCard(result, index));
  });

  sourcePanel.hidden = claimResults.length === 0;

  sourceHint.textContent =
    "Select a claim above to highlight its evidence in the source.";

  renderSource(data.source_text, [], "supported");

  renderConflicts(data);

  // Use the corrected response when available.
  // Otherwise use the backend final text or uploaded input.
  const finalResponse = String(
    corrected ||
    data.final_text ||
    lastSubmittedResponse ||
    responseText.value ||
    ""
  ).trim();

  const correctionEyebrow =
    correctionPanel.querySelector(".correction-eyebrow");

  const correctionHeading =
    correctionPanel.querySelector("h2");

  if (corrected) {
    if (correctionEyebrow) {
      correctionEyebrow.textContent = "Revised for accuracy";
    }

    if (correctionHeading) {
      correctionHeading.textContent = "Corrected response";
    }
  } else {
    if (correctionEyebrow) {
      correctionEyebrow.textContent = "Verified output";
    }

    if (correctionHeading) {
      correctionHeading.textContent = "Final response";
    }
  }

  correctionText.textContent = finalResponse;

  // Always render the verification timeline, using the original
  // verification result when the backend has no correction iterations.
  renderTimeline(data.iterations, finalResponse, score);

  // Explicitly reveal the response panel whenever a response is available.
  correctionPanel.hidden = !finalResponse;

  allClearText.textContent = clearMessage(score);
  allClear.hidden = false;

  results.hidden = false;

  results.scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
};

const readError = async (res) => {
  let detail = `Server error: ${res.status}`;

  try {
    const body = await res.json();

    if (body.error) {
      detail = body.error;
    } else if (body.detail) {
      detail = JSON.stringify(body.detail);
    }
  } catch (_) {}

  return detail;
};

const runVerification = async () => {
  if (
    !responseText.value.trim() &&
    !responseFileInput.files[0]
  ) {
    alert("Please paste or upload the LLM response.");
    return;
  }

  if (
    !sourceText.value.trim() &&
    !sourceFileInput.files.length &&
    !(sourceUrls?.value || "").trim()
  ) {
    alert("Please paste or upload the source document.");
    return;
  }

  runBtn.disabled = true;
  results.hidden = true;
  statusRow.hidden = false;

  startStatusCycle();

  try {
    const formData = new FormData();

    formData.append("llm_response", responseText.value);
    formData.append("source_text", sourceText.value);
    formData.append("source_urls", sourceUrls?.value || "");
    formData.append(
      "unsupported_policy",
      policySelect.value
    );

    if (modeSelect.value) {
      formData.append("mode", modeSelect.value);
    }

    if (responseFileInput.files[0]) {
      formData.append(
        "response_file",
        responseFileInput.files[0]
      );
    }

    Array.from(sourceFileInput.files).forEach((file) => {
      formData.append("source_files", file);
    });

    // Save the actual submitted response, including uploaded files.
    lastSubmittedResponse = responseText.value.trim();

    if (responseFileInput.files[0]) {
      lastSubmittedResponse =
        await responseFileInput.files[0].text();
    }

    const res = await fetch("/api/check", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      throw new Error(await readError(res));
    }

    const data = await res.json();

    renderResults(data);

  } catch (err) {
    alert("Something went wrong: " + err.message);
    console.error(err);

  } finally {
    stopStatusCycle();

    statusRow.hidden = true;
    runBtn.disabled = false;
  }
};

const applyConfig = (config) => {
  if (!config.nli_available) {
    ["cascade", "full"].forEach((value) => {
      const option = modeSelect.querySelector(
        `option[value="${value}"]`
      );

      if (option && !option.disabled) {
        option.disabled = true;
        option.textContent += " — NLI not enabled";
      }
    });
  }

  const parts = [
    config.nli_available
      ? "NLI verifier on"
      : "LLM-only verification",
  ];

  parts.push(
    config.calibrated
      ? "calibrated confidence"
      : "uncalibrated confidence"
  );

  footerText.textContent =
    `Source-grounded verification · ${parts.join(" · ")}`;
};

const loadConfig = async () => {
  try {
    const res = await fetch("/api/config");

    if (res.ok) {
      applyConfig(await res.json());
    }
  } catch (_) {}
};

sourceFileInput.addEventListener("change", () => {
  sourceFilename.textContent = Array.from(sourceFileInput.files)
    .map((file) => file.name)
    .join(", ");
});

responseFileInput.addEventListener("change", () => {
  responseFilename.textContent =
    responseFileInput.files[0]
      ? responseFileInput.files[0].name
      : "";
});

runBtn.addEventListener("click", runVerification);

certificateDownload.addEventListener("click", () => {
  if (!lastData || !lastData.certificate) return;

  const payload = {
    ...lastData.certificate,
    verification_summary: lastData.score,
  };

  const blob = new Blob(
    [JSON.stringify(payload, null, 2)],
    { type: "application/json" }
  );

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = url;

  link.download =
    `verification-certificate-${new Date()
      .toISOString()
      .replace(/[:.]/g, "-")}.json`;

  document.body.appendChild(link);

  link.click();
  link.remove();

  URL.revokeObjectURL(url);
});

loadConfig();
