/**
 * Leopard 44 KB — local web UI client
 * No dependencies, no build step, no external imports.
 *
 * Responsibilities:
 *  - Scope toggle: persist to localStorage, reflect on buttons
 *  - Ask form submit: fetch POST /query, handle streaming SSE response
 *  - SPEC-SHAPED SSE parser: blank-line-delimited events, multi-data: lines
 *    joined with \n, streaming TextDecoder with {stream:true}, \r\n-safe
 *  - Source event: build card via DOM nodes (XSS-safe, no innerHTML of untrusted text)
 *  - Token event: append text nodes via textContent (NEVER innerHTML) — XSS-safe
 *  - Refusal / error / dropped-connection: explicit states, no hanging spinner
 *  - Done event: linkify [n] chips in settled answer; scroll-to-card anchoring
 */

/* ==========================================================================
   CONSTANTS
   ========================================================================== */

const SCOPE_KEY = 'l44-scope';
const DEFAULT_SCOPE = 'all';
const VALID_SCOPES = ['all', 'shared', 'vessel'];

/* ==========================================================================
   SCOPE TOGGLE — localStorage-persisted, aria-pressed-reflected
   ========================================================================== */

/**
 * Read the active scope from localStorage, or fall back to DEFAULT_SCOPE.
 * Returns one of: 'all' | 'shared' | 'vessel'.
 */
function getScope() {
  const stored = localStorage.getItem(SCOPE_KEY);
  return VALID_SCOPES.includes(stored) ? stored : DEFAULT_SCOPE;
}

/**
 * Write the active scope to localStorage and update all button aria-pressed states.
 * Does NOT re-run the previous query (UI-SPEC §4.2 SC3).
 */
function setScope(scope) {
  if (!VALID_SCOPES.includes(scope)) return;
  localStorage.setItem(SCOPE_KEY, scope);
  document.querySelectorAll('.scope-btn').forEach(btn => {
    btn.setAttribute('aria-pressed', btn.dataset.scope === scope ? 'true' : 'false');
  });
}

/**
 * Show the transient "Scope set to X" helper line for ~3s, then clear it.
 */
function showScopeHelper(scope) {
  const helper = document.getElementById('scope-helper');
  if (!helper) return;
  const label = scope.charAt(0).toUpperCase() + scope.slice(1);
  helper.textContent = `Scope set to ${label} — applies to your next question.`;
  clearTimeout(helper._timer);
  helper._timer = setTimeout(() => {
    helper.textContent = '';
  }, 3000);
}

/* ==========================================================================
   SPEC-SHAPED SSE PARSER
   Blank-line-delimited events. Multiple data: lines per event joined with \n.
   Streaming TextDecoder with {stream:true} for multibyte chars split across reads.
   \r\n and \n both supported.
   ========================================================================== */

/**
 * Parse a complete event block (lines separated by \n, no trailing blank line).
 * Returns { eventName: string, data: string }.
 * Multiple data: lines are joined with \n (SSE spec § 9.2.6).
 */
function flushEvent(block) {
  const lines = block.split('\n');
  let eventName = 'message';
  const dataLines = [];

  for (const line of lines) {
    if (line.startsWith('event:')) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      // data: value (leading space after colon is stripped per SSE spec)
      const value = line.slice(5);
      dataLines.push(value.startsWith(' ') ? value.slice(1) : value);
    }
    // id: and retry: fields are intentionally not processed (not used by the server)
  }

  const data = dataLines.join('\n');
  return { eventName, data };
}

/**
 * Incrementally parse the SSE stream buffer.
 * Splits on the blank-line event delimiter (\n\n).
 * Keeps the trailing partial event in the buffer (not yet complete).
 * Calls handleEvent(eventName, joinedData) for each complete event.
 *
 * @param {string} buffer  - accumulated decoded text (may contain partial events)
 * @param {Function} onEvent - called with (eventName: string, data: string)
 * @returns {string} - remaining buffer (partial trailing event, no blank line yet)
 */
function parseSSE(buffer, onEvent) {
  // Normalize \r\n → \n so the blank-line detection works uniformly
  buffer = buffer.replace(/\r\n/g, '\n');

  // Split on the blank-line event delimiter
  const parts = buffer.split('\n\n');

  // All but the last are complete events; the last is a partial (or empty)
  const completeEvents = parts.slice(0, -1);
  const remaining = parts[parts.length - 1];

  for (const block of completeEvents) {
    const trimmed = block.trim();
    if (trimmed.length > 0) {
      const { eventName, data } = flushEvent(trimmed);
      onEvent(eventName, data);
    }
  }

  return remaining;
}

/* ==========================================================================
   DOM HELPERS
   ========================================================================== */

/**
 * Escape HTML special characters for safe insertion into innerHTML.
 * Used ONLY in the "done" branch after all tokens have been accumulated.
 */
function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Return the layer CSS class string for a given layer value.
 * Falls back to 'unknown' for unrecognised values.
 */
function layerClass(layer) {
  return ['vessel', 'shared', 'community'].includes(layer) ? layer : 'unknown';
}

/**
 * Return an inline SVG glyph for the given layer (1.5px stroke, currentColor).
 * Non-colour-coded layer attribution (a11y).
 */
function layerGlyph(layer) {
  const cls = 'icon-layer';
  const svgOpen = `<svg class="${cls}" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">`;
  switch (layer) {
    case 'vessel':
      // Abstract hull + horizon line
      return svgOpen + '<path d="M3 17l4-8 5 4 5-6 4 10"/><line x1="2" y1="21" x2="22" y2="21"/></svg>';
    case 'shared':
      // Book / manual mark
      return svgOpen + '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>';
    case 'community':
      // People / group mark
      return svgOpen + '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>';
    default:
      // Circle-dash for unknown
      return svgOpen + '<circle cx="12" cy="12" r="10"/><line x1="8" y1="12" x2="16" y2="12"/></svg>';
  }
}

/* ==========================================================================
   SOURCE CARD BUILDER (DOM-only, no innerHTML of untrusted text)
   ========================================================================== */

/**
 * Build and append a source card to #sources-list from a parsed source event.
 * Reveals the "Sources" section heading on the first card.
 *
 * @param {{ n: number, layer: string, title: string, page_start: number|null, page_end: number|null }} source
 */
function appendSourceCard(source) {
  const sourcesList = document.getElementById('sources-list');
  const sourcesSection = document.getElementById('sources');
  if (!sourcesList) return;

  // Reveal the Sources section when the first card arrives (D-04)
  if (sourcesSection && sourcesSection.hidden) {
    sourcesSection.hidden = false;
  }

  const lc = layerClass(source.layer);

  // Build card via DOM — textContent for all untrusted string values (XSS-safe)
  const card = document.createElement('div');
  card.className = `source-card layer-${lc}`;
  card.id = `src-${source.n}`;

  // Card header: badge + title + index [n]
  const header = document.createElement('div');
  header.className = 'source-card-header';

  // Layer badge (uses innerHTML for the SVG glyph + label — SVG is hand-authored, label is escaped)
  const badge = document.createElement('span');
  badge.className = `badge layer-${lc}`;
  badge.innerHTML = layerGlyph(source.layer) + escapeHtml(lc.toUpperCase());
  header.appendChild(badge);

  // Title — textContent only (may be user-derived from KB document title)
  const title = document.createElement('span');
  title.className = 'source-card-title';
  title.textContent = source.title;
  header.appendChild(title);

  // Index [n] — right-aligned, mono
  const index = document.createElement('span');
  index.className = 'source-card-index';
  index.textContent = `[${source.n}]`;
  header.appendChild(index);

  card.appendChild(header);

  // Card meta: page ref (if present; page 0 is valid — check !== null, not truthiness)
  const meta = document.createElement('div');
  meta.className = 'source-card-meta';

  if (source.page_start !== null && source.page_start !== undefined) {
    const pageRef = document.createElement('span');
    pageRef.className = 'source-card-page';
    // Mono page ref: "p.47" or "p.47–49" when end differs
    const start = source.page_start;
    const end = source.page_end;
    const pageText = (end !== null && end !== undefined && end !== start)
      ? `p.${start}–${end}`
      : `p.${start}`;
    pageRef.textContent = pageText;
    meta.appendChild(pageRef);
  }

  card.appendChild(meta);

  // Option A: expandable cited passage — the exact retrieved chunk that grounded
  // the answer. Native <details> = keyboard-accessible, zero JS state. textContent
  // keeps it XSS-safe. Only shown when the source event carries content.
  if (source.content) {
    const disclosure = document.createElement('details');
    disclosure.className = 'source-card-passage-wrap';

    const summary = document.createElement('summary');
    summary.className = 'source-card-toggle';
    summary.textContent = 'Show the passage used';
    disclosure.appendChild(summary);

    // The chunk content already leads with its heading hierarchy (0d1f378), so we
    // don't render section_path separately — it's kept in the payload for future
    // deep-linking (Option B) but would just duplicate the passage's first line.

    const passage = document.createElement('div');
    passage.className = 'source-card-passage';
    passage.textContent = source.content;  // XSS-safe
    disclosure.appendChild(passage);

    card.appendChild(disclosure);
  }

  sourcesList.appendChild(card);
}

/* ==========================================================================
   ZONE HIGHLIGHT BUILDER (DOM-only, no innerHTML of untrusted text — VIS-02 / T-09-12)
   ========================================================================== */

/**
 * Build and insert a zone highlight block before the #sources section.
 * Called once per zone_highlight SSE event.
 *
 * D-12 graceful degradation: the <details> schematic control is rendered ONLY
 * when zone.geometry && zone.schematic_image. If geometry is null, name + cue
 * still render (client shows what the server sent; no schematic placeholder).
 *
 * VIS-02: all string values use textContent — NEVER innerHTML for untrusted text.
 * Amber SVG polygon: viewBox set to natural image pixel dims; polygon points in
 * those same dims so vector-effect="non-scaling-stroke" keeps the outline thin
 * regardless of CSS scaling (review concern 3).
 *
 * @param {{ zone_id: number, name: string, cue: string|null,
 *           schematic_image: string|null, geometry: number[][]|null }} zone
 */
function appendZoneHighlight(zone) {
  const block = document.createElement('div');
  block.className = 'zone-highlight';
  // Phase 11 / D-11: blue deviation variant — applied as a BEM modifier class.
  // kind === 'deviation' means the zone was resolved from a deviation chunk, not
  // an inventory item; the polygon and block header render in --deviation (blue)
  // to distinguish at a glance from the --accent (amber) inventory highlight.
  const isDeviation = zone.kind === 'deviation';
  if (isDeviation) {
    block.classList.add('zone-highlight--deviation');
  }

  // Zone name — textContent only (zone.name is untrusted DB content)
  const nameEl = document.createElement('p');
  nameEl.className = 'zone-highlight-name';
  nameEl.textContent = zone.name || '';
  block.appendChild(nameEl);

  // Zone cue (vertical_desc) — textContent only
  if (zone.cue) {
    const cueEl = document.createElement('p');
    cueEl.className = 'zone-highlight-cue';
    cueEl.textContent = zone.cue;
    block.appendChild(cueEl);
  }

  // Collapsible schematic reveal — only when geometry AND schematic_image (D-12)
  if (zone.geometry && zone.schematic_image) {
    const details = document.createElement('details');
    details.className = 'schematic-reveal';

    const summary = document.createElement('summary');
    summary.textContent = 'Show on schematic';
    details.appendChild(summary);

    const container = document.createElement('div');
    container.className = 'schematic-container';

    // Schematic image — src is a same-origin /schematic-image/ route (T-09-15)
    const img = document.createElement('img');
    img.className = 'schematic-img';
    img.src = '/schematic-image/' + encodeURIComponent(zone.schematic_image);
    img.alt = 'Schematic for ' + zone.name;
    container.appendChild(img);

    // SVG polygon overlay — created inline so we can set the viewBox dynamically.
    // Strategy: load the image to get its naturalWidth/naturalHeight, set the SVG
    // viewBox to those pixel dims, then draw the polygon with non-scaling-stroke
    // so the outline stays thin as the image is CSS-scaled (review concern 3).
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'zone-polygon-overlay');
    svg.setAttribute('aria-hidden', 'true');

    const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    polygon.setAttribute('vector-effect', 'non-scaling-stroke');
    // Use --deviation (blue) for deviation highlights; --accent (amber) for inventory.
    // DOM-only: color var references are safe literals (VIS-02 / T-11-08).
    const polygonColor = isDeviation ? 'var(--deviation)' : 'var(--accent)';
    polygon.setAttribute('stroke', polygonColor);
    polygon.setAttribute('stroke-width', '2');
    polygon.setAttribute('fill', polygonColor);
    polygon.setAttribute('fill-opacity', '0.25');
    svg.appendChild(polygon);
    container.appendChild(svg);

    // Once the image loads, set the SVG viewBox to natural pixel dims and
    // map the normalized geometry coords to those dims.
    img.addEventListener('load', function () {
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      if (!w || !h) return;
      svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
      const points = zone.geometry.map(function (p) {
        return (p[0] * w) + ',' + (p[1] * h);
      }).join(' ');
      polygon.setAttribute('points', points);
    });

    details.appendChild(container);
    block.appendChild(details);
  }

  // Insert between answer area and sources section (before #sources)
  const sourcesSection = document.getElementById('sources');
  if (sourcesSection && sourcesSection.parentNode) {
    sourcesSection.parentNode.insertBefore(block, sourcesSection);
  } else {
    // Fallback: append to answer-area if sources not found
    const area = document.getElementById('answer-area');
    if (area) area.appendChild(block);
  }
}

/* ==========================================================================
   TOKEN INSERTION (textContent only — XSS-safe)
   ========================================================================== */

/** Get or create the .answer-text container inside #answer-area. */
function getAnswerTextEl() {
  const area = document.getElementById('answer-area');
  if (!area) return null;
  let textEl = area.querySelector('.answer-text');
  if (!textEl) {
    textEl = document.createElement('div');
    textEl.className = 'answer-text';
    area.appendChild(textEl);
  }
  return textEl;
}

/** Ensure a blinking caret element trails the answer text. */
function ensureCaret() {
  const area = document.getElementById('answer-area');
  if (!area) return;
  let caret = area.querySelector('.answer-caret');
  if (!caret) {
    caret = document.createElement('span');
    caret.className = 'answer-caret';
    caret.setAttribute('aria-hidden', 'true');
    area.appendChild(caret);
  }
  return caret;
}

/** Remove the streaming caret. */
function removeCaret() {
  const area = document.getElementById('answer-area');
  if (!area) return;
  const caret = area.querySelector('.answer-caret');
  if (caret) caret.remove();
}

/* ==========================================================================
   EVENT HANDLERS (one per SSE event name)
   ========================================================================== */

/**
 * "source" event — paint a source card.
 * Server JSON: { n, layer, title, page_start, page_end }
 */
function handleSource(data) {
  let source;
  try {
    source = JSON.parse(data);
  } catch (_) {
    return; // malformed — ignore
  }
  appendSourceCard(source);
}

/**
 * "token" event — append raw token text to the answer area via TEXT NODE.
 * NEVER use innerHTML for this — tokens from the LLM may contain < > & (XSS surface).
 * The CSS pre-wrap rule renders \n as paragraph breaks.
 */
function handleToken(data) {
  const textEl = getAnswerTextEl();
  if (!textEl) return;
  // Append as a DOM text node — not innerHTML, not textContent +=
  textEl.appendChild(document.createTextNode(data));
  ensureCaret();
}

/**
 * "refusal" event — calm "no-match" state (NOT an error).
 * Render the server-supplied message verbatim in --refusal colour.
 * No sources section, no spinner.
 */
function handleRefusal(data) {
  removeCaret();
  const area = document.getElementById('answer-area');
  if (!area) return;

  area.innerHTML = ''; // clear any partial content

  const wrap = document.createElement('div');
  wrap.className = 'answer-refusal';

  const msg = document.createElement('p');
  msg.textContent = data; // verbatim server message, via textContent
  wrap.appendChild(msg);

  const tip = document.createElement('p');
  tip.className = 'refusal-tip';
  tip.textContent = 'Tip: add sources with the CLI (l44 ingest), or rephrase your question.';
  wrap.appendChild(tip);

  area.appendChild(wrap);
}

/**
 * "error" event — Ollama-down or other server RuntimeError.
 * Render a distinct fault block with the server message + a retry affordance.
 */
function handleError(data, retryFn) {
  removeCaret();
  const area = document.getElementById('answer-area');
  if (!area) return;

  area.innerHTML = ''; // clear partial streaming content

  const block = document.createElement('div');
  block.className = 'fault-block';

  // Fault heading with warning SVG glyph
  const heading = document.createElement('div');
  heading.className = 'fault-heading';
  heading.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
  const headingText = document.createTextNode(' The local AI model isn’t running.');
  heading.appendChild(headingText);
  block.appendChild(heading);

  // Server message verbatim in mono
  const message = document.createElement('pre');
  message.className = 'fault-message';
  message.textContent = data; // textContent — verbatim, no HTML injection
  block.appendChild(message);

  // Retry affordance
  const retryBtn = document.createElement('button');
  retryBtn.type = 'button';
  retryBtn.className = 'fault-retry';
  retryBtn.textContent = 'Try again';
  retryBtn.addEventListener('click', retryFn);
  block.appendChild(retryBtn);

  area.appendChild(block);
}

/**
 * "done" event — stream complete.
 * Removes caret, re-enables input, linkifies [n] citation chips in the settled answer.
 *
 * @param {string} data  - JSON { bad_citations: [int, ...] }
 * @param {Function} reenableFn - re-enables the form inputs
 */
function handleDone(data, reenableFn) {
  removeCaret();
  reenableFn();

  let badCitations = [];
  try {
    const parsed = JSON.parse(data);
    badCitations = Array.isArray(parsed.bad_citations) ? parsed.bad_citations : [];
  } catch (_) {
    // malformed done data — proceed with empty bad list
  }

  // Linkify [n] in the settled answer text.
  // Strategy: collect accumulated text, escape for HTML, replace [n] with cite-chip
  // buttons ONLY when #src-{n} exists (un-fakeable citations — C5).
  const textEl = getAnswerTextEl();
  if (!textEl) return;

  const rawText = textEl.textContent;

  // Escape the full text, then replace [n] patterns with chip buttons (or plain text)
  const escapedText = escapeHtml(rawText);
  const linkedHtml = escapedText.replace(/\[(\d+)\]/g, (match, nStr) => {
    const n = parseInt(nStr, 10);
    const card = document.getElementById(`src-${n}`);
    // Only linkify if: card exists AND n is not in bad_citations
    if (card && !badCitations.includes(n)) {
      return `<button type="button" class="cite-chip" data-citation="${n}">[${n}]</button>`;
    }
    // No card or bad citation — render as plain escaped text
    return match; // match is already the escaped form (digits only, safe)
  });

  // Set innerHTML on the container — safe because we escaped & < > before replacement,
  // and the only injected markup is our own cite-chip buttons (no user-derived HTML).
  textEl.innerHTML = linkedHtml;

  // Wire cite-chip click handlers
  textEl.querySelectorAll('.cite-chip[data-citation]').forEach(chip => {
    const n = chip.dataset.citation;
    chip.addEventListener('click', () => scrollToCard(n));
  });
}

/**
 * Scroll to the source card #src-{n} and apply a brief highlight pulse.
 */
function scrollToCard(n) {
  const card = document.getElementById(`src-${n}`);
  if (!card) return;
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  // Brief highlight pulse (~600ms) — reduced-motion guard collapses this via CSS
  card.classList.add('just-jumped');
  setTimeout(() => card.classList.remove('just-jumped'), 600);
}

/**
 * Dispatch to the correct handler based on SSE event name.
 * Called once per complete event block by the stream pump.
 */
function handleEvent(eventName, data, retryFn, reenableFn, flags) {
  switch (eventName) {
    case 'source':
      handleSource(data);
      break;
    case 'token':
      handleToken(data);
      break;
    case 'refusal':
      flags.sawTerminal = true;
      handleRefusal(data);
      break;
    case 'error':
      flags.sawTerminal = true;
      handleError(data, retryFn);
      break;
    case 'zone_highlight': {
      // VIS-01: textual-first amber zone block (D-10/D-11/D-12).
      // Emitted by server AFTER last source event and BEFORE done (review concern 1).
      let zone;
      try { zone = JSON.parse(data); } catch (_) { break; }
      appendZoneHighlight(zone);
      break;
    }
    case 'done':
      flags.sawTerminal = true;
      handleDone(data, reenableFn);
      break;
    default:
      // Unknown event type — ignore gracefully
      break;
  }
}

/* ==========================================================================
   FORM — submit handler + state management
   ========================================================================== */

/** Disable the ask form while a query is in flight. */
function disableForm(form) {
  const textarea = form.querySelector('#question');
  const btn = form.querySelector('#ask-btn');
  if (textarea) textarea.disabled = true;
  if (btn) {
    btn.disabled = true;
    btn.classList.add('working');
    const label = btn.querySelector('.ask-btn-label');
    if (label) label.textContent = 'Asking…';
  }
}

/** Re-enable the ask form after a query completes or errors. */
function enableForm(form) {
  const textarea = form.querySelector('#question');
  const btn = form.querySelector('#ask-btn');
  if (textarea) textarea.disabled = false;
  if (btn) {
    btn.disabled = false;
    btn.classList.remove('working');
    const label = btn.querySelector('.ask-btn-label');
    if (label) label.textContent = 'Ask';
  }
}

/** Clear the answer area and sources list in preparation for a new query. */
function resetAnswerArea() {
  const area = document.getElementById('answer-area');
  const sourcesList = document.getElementById('sources-list');
  const sourcesSection = document.getElementById('sources');
  if (area) area.innerHTML = '';
  if (sourcesList) sourcesList.innerHTML = '';
  if (sourcesSection) sourcesSection.hidden = true;
}

/** Hide the empty-state example chips when a query is submitted. */
function hideEmptyState() {
  const es = document.getElementById('empty-state');
  if (es) es.style.display = 'none';
}

/**
 * Show a "Connection interrupted" notice (DISTINCT from the settled-answer Done state).
 * Muted, italic, with a retry affordance. Only shown when no terminal event arrived.
 */
function showDroppedNotice(retryFn) {
  const area = document.getElementById('answer-area');
  if (!area) return;

  const notice = document.createElement('div');
  notice.className = 'dropped-notice';

  // Warning glyph
  const glyph = document.createElement('span');
  glyph.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
  glyph.setAttribute('aria-hidden', 'true');
  notice.appendChild(glyph);

  const msgWrap = document.createElement('span');
  const msg = document.createTextNode('Connection interrupted — the answer may be incomplete.');
  msgWrap.appendChild(msg);
  notice.appendChild(msgWrap);

  const retryBtn = document.createElement('button');
  retryBtn.type = 'button';
  retryBtn.className = 'dropped-retry';
  retryBtn.textContent = 'Try again';
  retryBtn.addEventListener('click', retryFn);
  notice.appendChild(retryBtn);

  area.appendChild(notice);
}

/* ==========================================================================
   MAIN QUERY RUNNER — fetch POST /query + SPEC-SHAPED SSE pump
   ========================================================================== */

/**
 * Execute a query against POST /query, consuming the SSE response stream.
 *
 * SSE wire-format requirements (review HIGH — both reviewers):
 * - Events are blank-line-delimited (\n\n)
 * - Multiple data: lines per event → joined with \n (SSE spec; needed for tokens
 *   containing newlines / paragraph breaks that stream_generate emits as multi-line)
 * - TextDecoder with {stream:true} so multibyte UTF-8 chars split across reads survive
 * - \r\n normalised to \n
 *
 * @param {string} question - trimmed question text
 * @param {string} layer    - 'all' | 'shared' | 'vessel'
 * @param {HTMLFormElement} form
 */
async function runQuery(question, layer, form) {
  resetAnswerArea();
  disableForm(form);

  // Mutable flags shared across async callbacks
  const flags = { sawTerminal: false };

  const retryFn = () => {
    enableForm(form);
    runQuery(question, layer, form);
  };

  const reenableFn = () => enableForm(form);

  let response;
  try {
    response = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, layer, top_k: 5 }),
    });
  } catch (networkErr) {
    // Network-level failure (no response at all)
    handleError(`Network error: ${String(networkErr)}`, retryFn);
    enableForm(form);
    return;
  }

  if (!response.ok) {
    handleError(`Server returned ${response.status}`, retryFn);
    enableForm(form);
    return;
  }

  // SPEC-SHAPED streaming SSE parse
  const reader = response.body.getReader();
  // {stream: true} keeps a split multibyte UTF-8 char pending until the next read (review HIGH)
  const decoder = new TextDecoder('utf-8', { ignoreBOM: true });
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        // Stream ended. Flush any remaining buffer that forms a complete event
        // (server may omit the final blank line — be defensive).
        if (buffer.trim().length > 0) {
          // Normalize and attempt to parse as a final event block
          const normalized = buffer.replace(/\r\n/g, '\n').trim();
          if (normalized.length > 0) {
            const { eventName, data } = flushEvent(normalized);
            handleEvent(eventName, data, retryFn, reenableFn, flags);
          }
        }
        break;
      }

      // {stream:true} holds back a partial multibyte sequence until the next read
      buffer += decoder.decode(value, { stream: true });

      // Parse complete events out of the buffer; keep the trailing partial
      buffer = parseSSE(buffer, (eventName, data) => {
        handleEvent(eventName, data, retryFn, reenableFn, flags);
      });
    }

    // Flush any final partial buffer (TextDecoder may still hold bytes)
    const final = decoder.decode(); // flush the internal state
    if (final.length > 0) {
      buffer += final;
      buffer = parseSSE(buffer, (eventName, data) => {
        handleEvent(eventName, data, retryFn, reenableFn, flags);
      });
    }

  } catch (streamErr) {
    // Reader threw mid-stream (connection dropped)
    removeCaret();
    enableForm(form);
    if (!flags.sawTerminal) {
      showDroppedNotice(retryFn);
    }
    return;
  }

  // Stream ended cleanly but no terminal event arrived (unexpected — show dropped notice)
  if (!flags.sawTerminal) {
    removeCaret();
    enableForm(form);
    showDroppedNotice(retryFn);
  }
}

/* ==========================================================================
   INITIALISATION — wire up after DOM content loaded
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {

  // --- Scope toggle setup ---
  const activeScope = getScope();
  setScope(activeScope); // reflect persisted scope on buttons immediately

  document.querySelectorAll('.scope-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const scope = btn.dataset.scope;
      if (scope) {
        setScope(scope);
        showScopeHelper(scope);
      }
    });
  });

  // Keyboard nav within the scope toggle group (arrow keys move focus)
  const scopeToggle = document.querySelector('.scope-toggle');
  if (scopeToggle) {
    scopeToggle.addEventListener('keydown', e => {
      const buttons = Array.from(scopeToggle.querySelectorAll('.scope-btn'));
      const idx = buttons.indexOf(document.activeElement);
      if (idx === -1) return;
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        buttons[(idx + 1) % buttons.length].focus();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        buttons[(idx - 1 + buttons.length) % buttons.length].focus();
      } else if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        buttons[idx].click();
      }
    });
  }

  // --- Ask form setup ---
  const form = document.getElementById('ask-form');
  if (!form) return;

  // Example question chips populate the textarea on click
  document.querySelectorAll('.example-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const textarea = document.getElementById('question');
      if (textarea) {
        textarea.value = chip.textContent.trim();
        textarea.focus();
      }
    });
  });

  // Enter submits (Shift+Enter inserts a newline)
  const textarea = document.getElementById('question');
  if (textarea) {
    textarea.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        form.requestSubmit();
      }
    });
  }

  // Form submit handler
  form.addEventListener('submit', e => {
    e.preventDefault();

    const questionInput = document.getElementById('question');
    const question = questionInput ? questionInput.value.trim() : '';

    // Empty/whitespace-only question — do nothing (mirrors engine's empty guard)
    if (question.length === 0) return;

    hideEmptyState();

    const layer = getScope();
    runQuery(question, layer, form);
  });
});
