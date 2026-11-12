"use client";
import { useEffect, useRef, useState } from "react";
import {
  api,
  date,
  Details,
  Empty,
  Item,
  Pager,
  sourceName,
  useLoad,
  useWork,
} from "./shared";
import { artifactLink } from "./guideModel";
export default function Collection() {
  const { run, busy } = useWork();
  const status = useLoad("/collection");
  const [offset, setOffset] = useState(0),
    [source, setSource] = useState("");
  const records = useLoad(
    `/records?state=all&order=newest&offset=${offset}&source=${encodeURIComponent(source)}`,
  );
  const [selected, setSelected] = useState<Item | null>(null),
    [note, setNote] = useState(""),
    [group, setGroup] = useState(""),
    [automatic, setAutomatic] = useState(false);
  const noteId = useRef<{ id: string; timestamp: number } | null>(null),
    importAction = useRef<() => Promise<void>>(async () => {});
  importAction.current = async () => {
    const ok = await run(() => api("/import?limit=500"), {
      text: "Saved inputs checked. Any newly imported records are unreviewed; inspect the counts below.",
      action: { href: "/labeling", label: "Review the labeling queue" },
    });
    if (!ok) setAutomatic(false);
  };
  useEffect(() => {
    if (!automatic) return;
    const timer = setInterval(() => importAction.current(), 10000);
    return () => clearInterval(timer);
  }, [automatic]);
  return (
    <>
      <section className="ws-next">
        <div className="ws-toolbar">
          <div>
            <div className="eyebrow">Saved-input collection</div>
            <h2>
              {automatic ? "Watching for saved inputs" : "Ready when you are"}
            </h2>
          </div>
          <span className="ws-pill">
            {automatic ? "● Checking every 10 seconds" : "Paused"}
          </span>
        </div>
        <p>
          Bring saved phone, browser, and teaching records into review.
          Originals stay in capture storage.
        </p>
        <div className="ws-toolbar">
          <button disabled={busy} onClick={() => importAction.current()}>
            Collect saved inputs now
          </button>
          <button
            className="secondary"
            disabled={busy && !automatic}
            onClick={() => {
              setAutomatic(!automatic);
              if (!automatic) importAction.current();
            }}
          >
            {automatic
              ? "Pause automatic collection"
              : "Start automatic collection"}
          </button>
          <a href="/capture">Record camera, microphone, or file →</a>
        </div>
        <small>
          Automatic collection runs while this page is open. Pausing finishes
          the current batch; it does not stop phone or browser recording.
        </small>
      </section>
      <div className="ws-metrics">
        {[
          [status?.counts.raw || 0, "Collected records"],
          [status?.awaiting_review || 0, "Awaiting review"],
          [status?.counts.failure || 0, "Pipeline errors · all time"],
          [
            status
              ? (status.ledger_bytes / 1024 / 1024).toFixed(1) + " MB"
              : "—",
            "Pipeline storage",
          ],
        ].map(([n, label]) => (
          <div key={label}>
            <strong>{n}</strong>
            <span>{label}</span>
          </div>
        ))}
      </div>
      {status?.source_error && (
        <p role="alert" className="error">
          Saved-input storage unavailable: {status.source_error}.{" "}
          <a href="/capture">Open capture to save an experience</a>, then use
          Collect saved inputs now again. You can also add a note below.
        </p>
      )}
      <form
        className="ws-card"
        onSubmit={(e) => {
          e.preventDefault();
          const pending = noteId.current || {
            id: `note:${crypto.randomUUID()}`,
            timestamp: Date.now() / 1000,
          };
          noteId.current = pending;
          run(
            async () => {
              const id = pending.id;
              await api("/collect", {
                id,
                source_id: group.trim() || id,
                source_kind: "user-note",
                source_timestamp: pending.timestamp,
                group_ids: [group.trim() || id],
                origin: "human",
                payload: { text: note.trim() },
              });
              noteId.current = null;
              setNote("");
            },
            {
              text: "Note collected and saved as unreviewed evidence.",
              action: {
                href: artifactLink("labeling", pending.id),
                label: "Review this note",
              },
            },
          );
        }}
      >
        <h2>Collect a note</h2>
        <label>
          What did you observe?
          <textarea
            value={note}
            required
            onChange={(e) => {
              setNote(e.target.value);
              noteId.current = null;
            }}
            placeholder="Describe the evidence in your own words."
          />
        </label>
        <details>
          <summary>Group related notes (optional)</summary>
          <label>
            Session or document name (optional)
            <input
              value={group}
              onChange={(e) => {
                setGroup(e.target.value);
                noteId.current = null;
              }}
              placeholder="Keep related notes together"
            />
          </label>
        </details>
        <button disabled={busy || !note.trim()}>Add to labeling queue</button>
        {!note.trim() && (
          <small className="gm-reason">
            Write what you observed to add a note.
          </small>
        )}
        <p className="muted">
          A collected note is unreviewed. It becomes eligible for datasets only
          after explicit review.
        </p>
      </form>
      <section className="ws-card">
        <details>
          <summary>Collection sources and import settings</summary>
          <p className="muted">
            These switches control saved-input import. Capture sensors are
            controlled in the recording app.
          </p>
          {status?.sources.map((s: Item) => (
            <label className="ws-check" key={s.kind}>
              <input
                type="checkbox"
                checked={s.enabled}
                disabled={busy || automatic}
                onChange={(e) =>
                  run(
                    () =>
                      api("/collection/settings", {
                        sources: status.sources
                          .filter((v: Item) =>
                            v.kind === s.kind ? e.target.checked : v.enabled,
                          )
                          .map((v: Item) => v.kind),
                      }),
                    "Source preferences saved.",
                  )
                }
              />
              <span>
                <strong>{sourceName(s.kind)}</strong>
                <small>
                  {s.available
                    ? `${s.pending} saved inputs not yet imported`
                    : "No saved inputs yet"}
                </small>
              </span>
            </label>
          ))}
          <small>
            Pause automatic collection to change sources. Storage above excludes
            original media.
          </small>
        </details>
      </section>
      <section className="ws-card">
        <div className="ws-toolbar">
          <h2>Recent records</h2>
          <label>
            Source
            <select
              value={source}
              onChange={(e) => {
                setSource(e.target.value);
                setOffset(0);
              }}
            >
              <option value="">All sources</option>
              {[
                "user-note",
                "observations",
                "world_observations",
                "world_teachings",
                "utterances",
              ].map((s) => (
                <option key={s} value={s}>
                  {sourceName(s)}
                </option>
              ))}
            </select>
          </label>
          <a href="/labeling">Open labeling queue →</a>
        </div>
        <div className="ws-list">
          {records?.items.map((r: Item) => (
            <button
              key={r.id}
              className="ws-record"
              onClick={() =>
                run(async () =>
                  setSelected(
                    await api("/artifact?id=" + encodeURIComponent(r.id)),
                  ),
                )
              }
            >
              <strong>{r.preview || sourceName(r.source_kind)}</strong>
              <span>
                {sourceName(r.source_kind)} · {date(r.source_timestamp)}
              </span>
              <small>
                {r.annotation?.reviewed ? "Human reviewed" : "Awaiting review"}{" "}
                · Received in pipeline
              </small>
            </button>
          ))}
        </div>
        {records?.total === 0 && (
          <Empty>
            No records here yet. Collect saved inputs or add a note above.
          </Empty>
        )}
        <Pager
          offset={offset}
          total={records?.total || 0}
          onChange={setOffset}
        />
        {selected && (
          <div className="ws-inspect">
            <h3>Original record</h3>
            <p>
              {selected.payload.text ||
                "This record contains structured sensor or teaching evidence."}
            </p>
            <p>
              Origin: {selected.origin} · Source: {selected.source_id}
            </p>
            <a href={"/labeling?record=" + encodeURIComponent(selected.id)}>
              Review this record →
            </a>
            <Details
              value={selected}
              title="Source provenance and original payload"
            />
          </div>
        )}
      </section>
      <details className="gm-disclosure">
        <summary>Collection history and past errors</summary>
        <div className="ws-columns">
          <section className="ws-card">
            <h2>Collection history</h2>
            {status?.recent_runs.length === 0 && (
              <Empty>No imports have run yet.</Empty>
            )}
            {status?.recent_runs.map((r: Item) => (
              <p key={r.id}>
                {r.imported} records imported ·{" "}
                {r.more_may_remain
                  ? "More may remain; collect again"
                  : "Batch complete"}
                <br />
                <small>{r.sources.map(sourceName).join(", ")}</small>
              </p>
            ))}
            <Details
              value={status?.activity}
              title="Recent pipeline activity"
            />
          </section>
          <section className="ws-card">
            <h2>Recent pipeline errors</h2>
            {status?.recent_failures.length === 0 && (
              <Empty>No pipeline errors recorded.</Empty>
            )}
            {status?.recent_failures.map((f: Item) => (
              <div className="ws-inspect" key={f.id}>
                <strong>{f.stage}</strong>
                <p>{f.message}</p>
                <small>
                  History is retained after retry. Correct the input or
                  connection, then retry in that workspace.
                </small>
              </div>
            ))}
            <p className="muted">
              Interrupted recordings and upload failures remain visible in
              capture history and the phone outbox.
            </p>
          </section>
        </div>
      </details>
    </>
  );
}
