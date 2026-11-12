"use client";
import { useEffect, useState } from "react";

type Item = Record<string, any>;
const stages = [
  ["raw", "1. Collected"],
  ["annotation", "2. Labels"],
  ["dataset", "3. Datasets"],
  ["experiment", "4. Candidates"],
  ["evaluation", "5. Evaluation"],
  ["deployment", "6. Deployment"],
  ["failure", "Failures"],
];
async function request(path: string, body?: unknown) {
  const response = await fetch("/api/v2/pipeline" + path, {
    method: body === undefined ? "GET" : "POST",
    headers:
      body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    if (response.status === 401) window.dispatchEvent(new Event("gm-unpaired"));
    throw new Error(await response.text());
  }
  return response.json();
}

export default function PipelinePanel({ enabled }: { enabled: boolean }) {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<Item | null>(null);
  const [kind, setKind] = useState("raw");
  const [offset, setOffset] = useState(0);
  const [items, setItems] = useState<Item[]>([]);
  const [selected, setSelected] = useState<Item | null>(null);
  const [text, setText] = useState("");
  const [facts, setFacts] = useState("");
  const [labelStatus, setLabelStatus] = useState("current");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState<Item | null>(null);

  async function refresh() {
    const [nextStatus, nextItems] = await Promise.all([
      request("/status"),
      request(`/items/${kind}?offset=${offset}`),
    ]);
    setStatus(nextStatus);
    setItems(nextItems);
  }
  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await action();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  useEffect(() => {
    if (!enabled || !open) return;
    let cancelled = false;
    Promise.all([
      request("/status"),
      request(`/items/${kind}?offset=${offset}`),
    ])
      .then(([s, rows]) => {
        if (!cancelled) {
          setStatus(s);
          setItems(rows);
        }
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [enabled, open, kind, offset]);
  if (!enabled) return null;
  return (
    <details
      className="card"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>
        <strong>Collection, review & experiments</strong>
      </summary>
      <p>
        Saved sources, editable labels, fixed datasets and evaluated recall
        configurations. The live world learner continues separately.
      </p>
      {error && <p role="alert">{error}</p>}
      <div className="row">
        <button
          disabled={busy}
          onClick={() =>
            run(async () => {
              await request("/import", {});
              await refresh();
            })
          }
        >
          Collect next 500 saved inputs
        </button>
        <button disabled={busy} onClick={() => run(refresh)}>
          Refresh
        </button>
        <span>{status?.awaiting_review ?? "—"} awaiting review</span>
      </div>
      <p>
        Deployed recall:{" "}
        {status?.deployed
          ? `${status.deployed.config.algorithm} · ${status.deployed.synthetic ? "synthetic benchmark only" : "reviewed local data"}`
          : "none"}
        . No automatic training or deployment.
      </p>
      <div className="row" role="group" aria-label="Pipeline stages">
        {stages.map(([value, name]) => (
          <button
            key={value}
            aria-pressed={kind === value}
            onClick={() => {
              setKind(value);
              setOffset(0);
              setSelected(null);
            }}
          >
            {name} ({status?.counts?.[value] ?? 0})
          </button>
        ))}
      </div>
      {items.length === 0 && (
        <p className="muted">No records in this stage yet.</p>
      )}
      {items.map((item) => (
        <details key={item.id}>
          <summary>
            {item.name ??
              item.source_kind ??
              item.stage ??
              item.action ??
              item.split ??
              item.config?.algorithm ??
              "Record"}{" "}
            · <small>{item.id}</small>
          </summary>
          <pre
            style={{
              overflow: "auto",
              maxHeight: 240,
              whiteSpace: "pre-wrap",
              overflowWrap: "anywhere",
            }}
          >
            {JSON.stringify(item, null, 2)}
          </pre>
          {kind === "raw" && (
            <button
              disabled={busy}
              onClick={() =>
                run(async () => {
                  const raw = await request(
                    "/artifact?id=" + encodeURIComponent(item.id),
                  );
                  setSelected({ ...raw, annotation: item.annotation });
                  setText(item.annotation?.text ?? raw.payload.text ?? "");
                  setFacts(item.annotation?.fact_keys?.join(", ") ?? "");
                  setLabelStatus(item.annotation?.status ?? "current");
                })
              }
            >
              Inspect source & review label
            </button>
          )}
        </details>
      ))}
      <div className="row">
        <button
          disabled={busy || offset === 0}
          onClick={() => setOffset(Math.max(0, offset - 25))}
        >
          Previous
        </button>
        <button
          disabled={busy || items.length < 25}
          onClick={() => setOffset(offset + 25)}
        >
          Next
        </button>
      </div>
      {selected && (
        <fieldset disabled={busy}>
          <legend>Review source {selected.id}</legend>
          <details>
            <summary>Original collected payload</summary>
            <pre style={{ overflow: "auto", maxHeight: 240 }}>
              {JSON.stringify(selected.payload, null, 2)}
            </pre>
          </details>
          <label>
            Memory statement
            <textarea value={text} onChange={(e) => setText(e.target.value)} />
          </label>
          <label>
            Underlying fact keys (comma separated)
            <input
              value={facts}
              onChange={(e) => setFacts(e.target.value)}
              placeholder="room:office:name"
            />
          </label>
          <label>
            Status
            <select
              value={labelStatus}
              onChange={(e) => setLabelStatus(e.target.value)}
            >
              {["current", "historical", "contested", "excluded"].map(
                (value) => (
                  <option key={value}>{value}</option>
                ),
              )}
            </select>
          </label>
          <p className="muted">
            Human review remains fallible. Use the CLI for validity dates and
            dataset preparation. Corrections invalidate snapshots that used the
            previous label.
          </p>
          <button
            onClick={() =>
              run(async () => {
                await request("/annotations", {
                  record_id: selected.id,
                  annotation: {
                    ...(selected.annotation
                      ? Object.fromEntries(
                          Object.entries(selected.annotation).filter(
                            ([key]) => !["id", "record_id"].includes(key),
                          ),
                        )
                      : {}),
                    actor: "paired-reviewer",
                    method: "human",
                    reviewed: true,
                    text,
                    fact_keys: facts
                      .split(",")
                      .map((value) => value.trim())
                      .filter(Boolean),
                    status: labelStatus,
                    previous_revision: selected.annotation?.id ?? null,
                  },
                });
                setSelected(null);
                await refresh();
              })
            }
          >
            Save reviewed revision
          </button>
          <button
            onClick={() =>
              run(async () => {
                await request("/policy", {
                  record_id: selected.id,
                  allowed: false,
                  actor: "paired-reviewer",
                });
                setSelected(null);
                await refresh();
              })
            }
          >
            Exclude from datasets & recall
          </button>
          <button onClick={() => setSelected(null)}>Close review</button>
        </fieldset>
      )}
      <label>
        Recall from the approved dataset
        <input value={query} onChange={(e) => setQuery(e.target.value)} />
      </label>
      <button
        disabled={busy || !query.trim() || !status?.deployed}
        onClick={() =>
          run(async () => {
            setAnswer(null);
            setAnswer(await request("/recall", { query }));
          })
        }
      >
        Recall with evidence
      </button>
      {answer && (
        <div>
          {answer.insufficient_evidence && <p>Insufficient evidence.</p>}
          {answer.uncertain && (
            <p>
              Conflicting or contested evidence; no single answer is
              established.
            </p>
          )}
          {answer.hits.map((hit: Item, i: number) => (
            <p key={i}>
              {hit.text}
              <br />
              <small>
                {hit.status} · Sources: {hit.evidence_ids.join(", ")}
              </small>
            </p>
          ))}
        </div>
      )}
      <p className="muted">
        Prepare datasets, freeze candidates, evaluate, promote and roll back
        with the pipeline CLI. Each stage and its results are listed above;
        failures remain available for inspection.
      </p>
    </details>
  );
}
