"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, Item, stages, Work, useLoad } from "./shared";
import Collection from "./Collection";
import Labeling from "./Labeling";
import { Datasets, Experiments, Evaluation, Deployment } from "./Stages";
import { Feedback, Guidance, HelpControls, Notice } from "./Guidance";
import { nextStep } from "./guideModel";
import "./workspaces.css";
function Overview() {
  const status = useLoad("/status");
  if (!status) return <p>Loading your pipeline…</p>;
  const counts = [
    status.counts.raw || 0,
    status.awaiting_review,
    status.counts.dataset || 0,
    status.counts.experiment || 0,
    status.counts.evaluation || 0,
    status.deployed ? 1 : 0,
  ];
  const { stage: next, reason } = nextStep(status);
  return (
    <>
      <section className="gm-goals">
        <h2>What would you like to do?</h2>
        <div className="gm-goals-grid">
          <article>
            <h3>Observe and teach</h3>
            <p>
              Use a camera or a recording. The live learner observes, asks
              questions, and revises its beliefs as you teach it. You can start
              with an unknown place.
            </p>
            <a className="ws-button" href="/capture">
              Start observing →
            </a>
          </article>
          <article>
            <h3>Review and test memories</h3>
            <p>
              Start with a note or saved capture evidence. Review it yourself,
              prepare questions, and compare search settings before approving
              reviewed recall.
            </p>
            <a className="ws-button" href="/collection">
              Start with Collection →
            </a>
          </article>
        </div>
        <p>
          These paths share saved evidence. Capture feeds the live learner as it
          happens; Collection imports saved inputs into the separate review
          workflow.
        </p>
      </section>
      <section className="ws-next">
        <div className="eyebrow">Suggested step for reviewed memories</div>
        <h2>{stages.find((s) => s[0] === next)?.[2]}</h2>
        <p>{reason}</p>
        <a className="ws-button" href={"/" + next}>
          Open {stages.find((s) => s[0] === next)?.[1]}
        </a>
      </section>
      <Guidance stage="overview" />
      <p className="muted">
        The counts below show available records and artifacts. They do not mean
        that a stage has passed or that a candidate is ready to deploy.
      </p>
      <div className="ws-stage-grid">
        {stages.slice(1).map(([key, title, description], i) => (
          <a key={key} className="ws-stage-card" href={"/" + key}>
            <span className="eyebrow">0{i + 1}</span>
            <h2>{title}</h2>
            <strong>{counts[i]}</strong>
            <p>
              {
                [
                  "records collected",
                  "awaiting review",
                  "prepared versions",
                  "frozen candidates",
                  "evaluation runs",
                  "active configuration",
                ][i]
              }
            </p>
            <span>{description} →</span>
          </a>
        ))}
      </div>
      <p className="muted">
        Collection → Labeling → Datasets → Experiments → Evaluation → Deployment
      </p>
      <section className="ws-card">
        <h2>The live learner has its own workspace</h2>
        <p>
          Observe, teach, inspect beliefs, import recordings, and control live
          capture using the existing sensing console.
        </p>
        <a href="/capture">Open capture & world learning →</a>
      </section>
    </>
  );
}
export default function Workspace({ stage = "overview" }: { stage?: string }) {
  const [paired, setPaired] = useState<boolean | null>(null),
    [code, setCode] = useState("");
  const [revision, setRevision] = useState(0),
    [busy, setBusy] = useState(false),
    [error, setError] = useState<{
      text: string;
      kind: "load" | "action";
    } | null>(null),
    [notice, setNotice] = useState<Notice | null>(null);
  const lock = useRef(false);
  const report = useCallback(
    (message: string, kind: "load" | "action" = "load") =>
      setError({ text: message, kind }),
    [],
  );
  const checkPairing = useCallback(() => {
    api("/api/status")
      .then((s) => setPaired(s.paired))
      .catch((e) => report(e.message));
  }, [report]);
  const run = async (fn: () => Promise<any>, message?: string | Notice) => {
    if (lock.current) return false;
    lock.current = true;
    setBusy(true);
    setError(null);
    try {
      await fn();
      if (message)
        setNotice(typeof message === "string" ? { text: message } : message);
      setRevision((x) => x + 1);
      return true;
    } catch (e) {
      report(e instanceof Error ? e.message : String(e), "action");
      return false;
    } finally {
      lock.current = false;
      setBusy(false);
    }
  };
  useEffect(() => {
    const lost = () => setPaired(false);
    window.addEventListener("gm-unpaired", lost);
    checkPairing();
    return () => window.removeEventListener("gm-unpaired", lost);
  }, []);
  const title = stages.find((s) => s[0] === stage);
  useEffect(() => {
    document.title = `Grounded Memory · ${title?.[1] || "Overview"}`;
  }, [stage]);
  return (
    <Work.Provider value={{ revision, busy, run, report }}>
      <div className="ws-app">
        <aside className="ws-nav">
          <a className="ws-brand" href="/">
            ◈{" "}
            <span>
              Grounded
              <br />
              Memory
            </span>
          </a>
          <div className="eyebrow">Your tools</div>
          <nav aria-label="Applications">
            {stages.map(([key, name], i) => (
              <a
                key={key}
                href={key === "overview" ? "/" : "/" + key}
                aria-current={stage === key ? "page" : undefined}
              >
                <span>{i === 0 ? "⌂" : `0${i}`}</span>
                {name}
              </a>
            ))}
          </nav>
          <a className="ws-capture" href="/capture">
            Capture & world learning ↗
          </a>
          <HelpControls />
          <p className="ws-local">
            ● Local to your Mac
            <br />
            <small>One shared evidence history</small>
          </p>
        </aside>
        <main className="ws-main">
          <header>
            <div>
              <div className="eyebrow">
                Grounded Memory {stage !== "overview" && " / " + title?.[1]}
              </div>
              <h1>{title?.[1]}</h1>
              <p className="muted">{title?.[2]}</p>
            </div>
            {paired && (
              <button
                className="secondary"
                disabled={busy}
                onClick={() => setRevision((x) => x + 1)}
              >
                Refresh
              </button>
            )}
          </header>
          {busy && (
            <p role="status">Working… wait for the result before continuing.</p>
          )}
          {error && (
            <div role="alert" className="error">
              <strong>
                {error.kind === "load"
                  ? "Could not load this information."
                  : "The action could not finish."}
              </strong>
              <p>{error.text}</p>
              {error.kind === "action" && (
                <p>
                  Check the message and correct the input or connection, then
                  use the original action button again. Review the current state
                  before retrying.
                </p>
              )}
              {error.kind === "load" && (
                <button
                  className="secondary"
                  onClick={() => {
                    setError(null);
                    checkPairing();
                    setRevision((x) => x + 1);
                  }}
                >
                  Retry loading
                </button>
              )}
            </div>
          )}
          {notice && (
            <Feedback notice={notice} dismiss={() => setNotice(null)} />
          )}
          {paired === null ? (
            <p>Connecting to your Mac…</p>
          ) : !paired ? (
            <form
              className="ws-card ws-pair"
              onSubmit={(e) => {
                e.preventDefault();
                run(async () => {
                  await api("/api/pair", { code });
                  setPaired(true);
                });
              }}
            >
              <h2>Connect once. Use every workspace.</h2>
              <p>
                Enter the pairing code printed when you start Grounded Memory.
              </p>
              <ol>
                <li>
                  On your Mac, run <code>make dev</code> in the Grounded Memory
                  folder and keep it running.
                </li>
                <li>
                  Enter the printed pairing code below. If it expired, restart
                  the service and use the new code.
                </li>
                <li>
                  Choose Observe and teach or Review and test memories on the
                  overview.
                </li>
              </ol>
              <p className="muted">
                This connects this browser to your local Mac. Use the same
                hostname for all workspaces. Pair the native iPhone separately
                using <code>make phone</code>.
              </p>
              <label>
                Pairing code
                <input
                  value={code}
                  inputMode="numeric"
                  autoComplete="one-time-code"
                  onChange={(e) => setCode(e.target.value)}
                  required
                />
              </label>
              <button disabled={busy}>Connect this browser</button>
            </form>
          ) : (
            <>
              {stage !== "overview" && <Guidance stage={stage} />}
              {stage === "overview" && <Overview />}
              {stage === "collection" && <Collection />}
              {stage === "labeling" && <Labeling />}
              {stage === "datasets" && <Datasets />}
              {stage === "experiments" && <Experiments />}
              {stage === "evaluation" && <Evaluation />}
              {stage === "deployment" && <Deployment />}
            </>
          )}
        </main>
      </div>
    </Work.Provider>
  );
}
