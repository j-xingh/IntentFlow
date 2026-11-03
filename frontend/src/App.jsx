import { useMemo, useState } from "react";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const examples = [
  "My virtual card is not working.",
  "I need to verify my identity.",
  "Why was I charged a cash withdrawal fee?",
];

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function displayIntent(intent = "") {
  return intent.replaceAll("_", " ");
}

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeView, setActiveView] = useState("hero");
  const trimmedText = useMemo(() => text.trim(), [text]);

  async function classify(event) {
    event.preventDefault();
    if (!trimmedText || loading) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmedText, top_k: 3 }),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        const message = typeof body.detail === "string" ? body.detail : `Request failed (${response.status}).`;
        throw new Error(message);
      }
      if (!body.prediction || !Array.isArray(body.top_predictions)) {
        throw new Error("The service returned an unexpected response. Please try again.");
      }
      setResult(body);
    } catch (err) {
      const isNetworkError = err instanceof TypeError;
      setError(isNetworkError
        ? `IntentFlow could not connect to the classification service. Please try again in a moment.`
        : (err.message || "The message could not be classified. Please try again."));
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setText("");
    setResult(null);
    setError("");
  }

  function updateText(value) {
    setText(value.slice(0, 1000));
    setResult(null);
    setError("");
  }

  return (
    <div className="site-shell">
      <header className="topbar">
        <a className="wordmark" href="#top" aria-label="IntentFlow home">
          <span className="wordmark-symbol" aria-hidden="true"><i /><i /><i /></span>
          <span>intentflow<span className="wordmark-period">.</span></span>
        </a>
        <nav className="topnav" aria-label="Main navigation">
          <button type="button" className="nav-action" onClick={() => setActiveView("classifier")}>Classifier</button>
          <a href="#method" onClick={() => setActiveView("hero")}>How it works</a>
        </nav>
      </header>

      <main id="top">
        <section className={`experience-glass ${activeView === "classifier" ? "is-classifier" : "is-hero"}`} id="experience" aria-live="polite">
          {activeView === "hero" ? (
          <section className="intro" aria-labelledby="page-title">
          <div className="intro-copy">
            <p className="eyebrow"><span className="eyebrow-index">01</span> BANKING INTENT CLASSIFICATION</p>
            <h1 id="page-title">Understand what<br className="desktop-break" /> a message <em>means.</em></h1>
            <p className="intro-description">
              Analyse a banking message and uncover the customer intent behind it.
            </p>
            <button className="hero-cta" type="button" onClick={() => setActiveView("classifier")}>Explore the classifier</button>
          </div>
        </section>
          ) : (
          <>
        <button className="panel-back" type="button" onClick={() => setActiveView("hero")}>Back to overview</button>
        <section className="classifier" id="classifier" aria-label="Banking intent classifier">
          <div className="composer">
            <div className="section-heading">
              <div>
                <p className="section-index">INPUT <span> / 01</span></p>
                <h2>Customer message</h2>
              </div>
              <span className="step-number">01</span>
            </div>
            <form onSubmit={classify}>
              <label className="sr-only" htmlFor="customer-message">Enter a banking-related customer message</label>
              <textarea
                id="customer-message"
                value={text}
                onChange={(event) => updateText(event.target.value)}
                placeholder="For example, “My virtual card is not working.”"
                maxLength={1000}
                rows={5}
                aria-describedby="input-help character-count"
              />
              <div className="input-meta">
                <span id="input-help">Describe the issue in your own words.</span>
                <span id="character-count" aria-live="polite">{text.length}/1000</span>
              </div>
              <div className="examples">
                <p className="examples-heading">SAMPLE MESSAGES</p>
                {examples.map((example, index) => (
                  <button className="example-button" type="button" key={example} onClick={() => updateText(example)}>
                    <span className="example-index">0{index + 1}</span>
                    <span>{example}</span>
                    <span className="example-add" aria-hidden="true">+</span>
                  </button>
                ))}
              </div>
              <div className="form-actions">
                <button className="button button-primary" type="submit" disabled={!trimmedText || loading}>
                  {loading ? <><span className="button-spinner" aria-hidden="true" /> Classifying</> : "Classify message"}
                </button>
                <button className="button button-secondary" type="button" onClick={reset} disabled={loading || (!text && !result && !error)}>Clear</button>
              </div>
            </form>
          </div>

          <div className="results" aria-live="polite" aria-busy={loading}>
            <div className="section-heading results-heading">
              <div>
                <p className="section-index">OUTPUT <span> / 02</span></p>
                <h2>Predicted intent</h2>
              </div>
              <span className="step-number">02</span>
            </div>

            {!result && !loading && !error && (
              <div className="empty-state">
                <div className="empty-mark" aria-hidden="true"><span>IF</span></div>
                <h3>Awaiting your message</h3>
                <p>Your classification results will appear here.</p>
              </div>
            )}

            {loading && (
              <div className="loading-state">
                <div className="skeleton skeleton-title" />
                <div className="skeleton skeleton-line" />
                <div className="skeleton skeleton-line short" />
                <div className="skeleton skeleton-row" />
                <div className="skeleton skeleton-row" />
                <p>Analysing your message…</p>
              </div>
            )}

            {error && (
              <div className="error-state" role="alert">
                <p className="error-kicker">UNABLE TO COMPLETE</p>
                <h3>We couldn’t classify this message.</h3>
                <p>{error}</p>
                <button className="retry-button" type="button" onClick={classify} disabled={!trimmedText || loading}>Try again</button>
              </div>
            )}

            {result && (
              <div className="result-state">
                <p className="result-kicker">PREDICTED INTENT</p>
                <h3 className="intent-name">{displayIntent(result.prediction.intent)}</h3>
                <div className="score-row"><span>Confidence score</span><strong>{formatPercent(result.prediction.confidence)}</strong></div>
                <div className="score-track" role="meter" aria-label="Top prediction confidence score" aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.round(Number(result.prediction.confidence) * 100)}><span style={{ width: `${Math.min(100, Math.max(0, Number(result.prediction.confidence) * 100))}%` }} /></div>
                <div className="ranked-header"><span>OTHER MATCHES</span><span>SCORE</span></div>
                <ol className="ranked-list">
                  {result.top_predictions.slice(1).map((item) => (
                    <li className="ranked-item" key={item.intent}>
                      <span>{displayIntent(item.intent)}</span><strong>{formatPercent(item.confidence)}</strong>
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </div>
        </section>
          </>
          )}
        </section>

        <section className="method" id="method" aria-labelledby="method-title">
          <div className="method-intro">
            <p className="section-index">HOW IT WORKS</p>
            <h2 id="method-title">From a message<br />to an intent.</h2>
          </div>
          <div className="method-detail"><span>01</span><h3>Read</h3><p>Your message is prepared for classification.</p></div>
          <div className="method-detail"><span>02</span><h3>Classify</h3><p>The system identifies the customer intent.</p></div>
          <div className="method-detail"><span>03</span><h3>Compare</h3><p>Explore the leading intent matches.</p></div>
        </section>
      </main>

      <footer className="footer">
        <a className="footer-brand" href="#top">intentflow<span>.</span></a>
        <p>Banking message intelligence.</p>
        <nav aria-label="Legal"><a href="/terms.html">Terms of Service</a><a href="/privacy.html">Privacy Policy</a></nav>
      </footer>
    </div>
  );
}
