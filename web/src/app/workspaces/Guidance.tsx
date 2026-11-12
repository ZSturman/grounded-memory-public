"use client";
import { useEffect, useState } from "react";
import { guides, readHiddenTips, writeHiddenTips } from "./guideModel";
import "./guidance.css";

export type Notice = { text: string; action?: { href: string; label: string } };
export function Feedback({
  notice,
  dismiss,
}: {
  notice: Notice;
  dismiss: () => void;
}) {
  return (
    <div className="gm-feedback">
      <div role="status">
        <strong>{notice.text}</strong>
      </div>
      {notice.action && (
        <a href={notice.action.href}>{notice.action.label} →</a>
      )}
      <button
        type="button"
        className="secondary"
        onClick={dismiss}
        aria-label="Dismiss confirmation"
      >
        Dismiss
      </button>
    </div>
  );
}

export function HelpControls() {
  return (
    <div className="gm-help-controls">
      <button
        type="button"
        className="secondary"
        onClick={() => {
          window.dispatchEvent(new Event("gm-show-help"));
          document
            .querySelector(".gm-guide")
            ?.scrollIntoView({ block: "start" });
        }}
      >
        Help & steps
      </button>
      <button
        type="button"
        className="secondary"
        onClick={() => {
          try {
            writeHiddenTips(window.localStorage, []);
          } catch {}
          window.dispatchEvent(new Event("gm-reset-tips"));
        }}
      >
        Show tips again
      </button>
    </div>
  );
}

export function WorkflowMap({ stage }: { stage: string }) {
  return (
    <nav className="gm-workflow" aria-label="How the work connects">
      <a
        href="/capture"
        aria-current={stage === "capture" ? "page" : undefined}
      >
        Capture & teach
      </a>
      <span aria-hidden="true">→ saved inputs →</span>
      {Object.keys(guides)
        .filter((k) => k !== "capture" && k !== "overview")
        .map((key, i) => (
          <a
            key={key}
            href={"/" + key}
            aria-current={stage === key ? "page" : undefined}
          >
            <span>{i + 1}. </span>
            {key === "labeling"
              ? "Labeling"
              : key.charAt(0).toUpperCase() + key.slice(1)}
          </a>
        ))}
    </nav>
  );
}

export function Guidance({ stage }: { stage: string }) {
  const [hidden, setHidden] = useState<string[]>([]);
  const [ready, setReady] = useState(false);
  const guide = guides[stage];
  function update(id: string, hide: boolean) {
    setHidden((previous) => {
      const next = hide
        ? [...new Set([...previous, id])]
        : previous.filter((v) => v !== id);
      try {
        writeHiddenTips(window.localStorage, next);
      } catch {}
      return next;
    });
  }
  useEffect(() => {
    try {
      setHidden(readHiddenTips(window.localStorage));
    } catch {}
    setReady(true);
    const show = () => update("steps." + stage, false);
    const reset = () => {
      setHidden([]);
      try {
        writeHiddenTips(window.localStorage, []);
      } catch {}
    };
    const sync = () => {
      try {
        setHidden(readHiddenTips(window.localStorage));
      } catch {}
    };
    window.addEventListener("gm-show-help", show);
    window.addEventListener("gm-reset-tips", reset);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener("gm-show-help", show);
      window.removeEventListener("gm-reset-tips", reset);
      window.removeEventListener("storage", sync);
    };
  }, [stage]);
  if (!guide || !ready) return null;
  const stepsHidden = hidden.includes("steps." + stage);
  return (
    <section className="gm-guide" aria-label="Step-by-step guidance">
      <WorkflowMap stage={stage} />
      <div className="gm-guide-heading">
        <h2>{stepsHidden ? "Need a hand?" : "Your steps"}</h2>
        <button
          type="button"
          className="secondary"
          aria-expanded={!stepsHidden}
          onClick={() => update("steps." + stage, !stepsHidden)}
        >
          {stepsHidden ? "Help & steps" : "Hide steps"}
        </button>
      </div>
      {!stepsHidden && (
        <>
          <p className="gm-flow">
            <strong>Start with:</strong> {guide.input}
            <br />
            <strong>You’ll create:</strong> {guide.output}
          </p>
          <ol>
            {guide.steps.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>
          <p className="gm-completion">
            <strong>When you’re done:</strong> {guide.done}
          </p>
        </>
      )}
      {!hidden.includes("tip." + stage) && (
        <aside className="gm-tip" aria-label="Helpful tip">
          <p>
            <strong>Good to know.</strong> {guide.tip}
          </p>
          <button
            type="button"
            className="secondary"
            onClick={() => update("tip." + stage, true)}
            aria-label="Dismiss helpful tip"
          >
            Dismiss tip
          </button>
        </aside>
      )}
    </section>
  );
}
