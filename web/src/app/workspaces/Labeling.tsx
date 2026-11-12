"use client";
import { useEffect, useRef, useState } from "react";
import {
  api,
  date,
  Details,
  Empty,
  EvidenceMedia,
  Item,
  Pager,
  sourceName,
  useLoad,
  useWork,
} from "./shared";
const dateInput = (value: number | null) =>
  value == null ? "" : new Date(value * 1000).toISOString().replace("Z", "");
const dateValue = (value: string, original: number | null) =>
  value === dateInput(original)
    ? original
    : value
      ? Date.parse(value + "Z") / 1000
      : null;
export default function Labeling() {
  const { run, busy } = useWork();
  const status = useLoad("/status");
  const [state, setState] = useState("pending"),
    [order, setOrder] = useState("priority"),
    [source, setSource] = useState(""),
    [search, setSearch] = useState(""),
    [offset, setOffset] = useState(0);
  const queue = useLoad(
    `/records?state=${state}&order=${order}&source=${encodeURIComponent(source)}&search=${encodeURIComponent(search)}&offset=${offset}`,
  );
  const [selected, setSelected] = useState<Item | null>(null),
    [raw, setRaw] = useState<Item | null>(null),
    [text, setText] = useState(""),
    [facts, setFacts] = useState(""),
    [labelStatus, setLabelStatus] = useState("current"),
    [notes, setNotes] = useState(""),
    [from, setFrom] = useState(""),
    [until, setUntil] = useState("");
  const [checked, setChecked] = useState<string[]>([]),
    [dirty, setDirty] = useState(false);
  async function open(row: Item) {
    if (dirty && !window.confirm("Discard the unsaved changes to this review?"))
      return;
    await run(async () => {
      const snapshot = await api("/record?id=" + encodeURIComponent(row.id));
      const original = snapshot.raw;
      const annotation = await api("/suggest", { record_id: row.id });
      setRaw(original);
      setSelected({ ...row, allowed: snapshot.allowed, annotation });
      setText(annotation.text);
      setFacts(annotation.fact_keys.join(", "));
      setLabelStatus(annotation.status);
      setNotes(annotation.notes || "");
      setFrom(dateInput(annotation.valid_from));
      setUntil(dateInput(annotation.valid_until));
      setDirty(false);
    });
  }
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const view = params.get("view");
    if (
      view &&
      ["all", "excluded", "flagged", "skipped", "reviewed"].includes(view)
    )
      setState(view);
    const id = params.get("record");
    if (id) open({ id });
  }, []);
  useEffect(() => {
    const warn = (e: BeforeUnloadEvent) => {
      if (dirty) {
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);
  async function decision(action: string, ids = selected ? [selected.id] : []) {
    if (!ids.length) return;
    if (
      dirty &&
      !window.confirm("Discard the unsaved label edits and continue?")
    )
      return;
    await run(
      async () => {
        await api("/triage", {
          record_ids: ids,
          action,
          actor: "paired-reviewer",
          notes: action === "flag" ? "Needs closer review" : "",
        });
        setSelected(null);
        setRaw(null);
        setChecked([]);
        setDirty(false);
      },
      action === "skip"
        ? "Saved for later. Find it under Skipped."
        : "Queue updated. No labels were confirmed.",
    );
  }
  async function save() {
    if (!selected) return;
    await run(
      async () => {
        await api("/annotations", {
          record_id: selected.id,
          annotation: {
            actor: "paired-reviewer",
            method: "human",
            reviewed: true,
            text,
            fact_keys: facts
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean),
            status: labelStatus,
            notes,
            valid_from: dateValue(from, selected.annotation.valid_from),
            valid_until: dateValue(until, selected.annotation.valid_until),
            previous_revision: selected.annotation?.id ?? null,
          },
        });
        setDirty(false);
        setSelected(null);
        setRaw(null);
        await api("/triage", {
          record_ids: [selected.id],
          action: "clear",
          actor: "paired-reviewer",
        });
      },
      {
        text: "Human review saved. The original record and previous revisions are preserved. Continue reviewing, or inspect eligible evidence in Datasets.",
        action: { href: "/datasets", label: "Prepare a dataset" },
      },
    );
  }
  const actions = useRef({ save, decision, open, queue, selected });
  actions.current = { save, decision, open, queue, selected };
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      const a = actions.current;
      if (
        busy ||
        e.altKey ||
        e.ctrlKey ||
        e.metaKey ||
        (e.target as HTMLElement)?.closest(
          "input,textarea,select,button,[contenteditable=true]",
        )
      )
        return;
      if (e.key === "s" && a.selected) {
        e.preventDefault();
        a.save();
      }
      if (e.key === "k" && a.selected) a.decision("skip");
      if (e.key === "f" && a.selected) a.decision("flag");
      if (["j", "n", "p"].includes(e.key) && a.queue?.items.length) {
        const index = a.queue.items.findIndex(
          (r: Item) => r.id === a.selected?.id,
        );
        a.open(
          a.queue.items[
            (index + (e.key === "p" ? -1 : 1) + a.queue.items.length) %
              a.queue.items.length
          ],
        );
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [busy]);
  useEffect(() => {
    setOffset(0);
    setChecked([]);
  }, [state, source, search, order]);
  return (
    <>
      <section className="ws-next">
        <div className="eyebrow">Up next</div>
        <h2>{status?.awaiting_review || 0} records need a human decision</h2>
        <p>
          Read the source, correct the proposed memory, and name the fact it
          describes. Uncertain? Save it for later.
        </p>
        <progress
          aria-label="Review progress"
          value={status?.reviewed || 0}
          max={Math.max(status?.counts.raw || 0, 1)}
        />
        <small>
          {status?.reviewed || 0} reviewed of {status?.counts.raw || 0}{" "}
          collected
        </small>
        <details>
          <summary>Keyboard shortcuts</summary>
          <p>
            Outside fields: J / N next · P previous · S save · K skip · F flag.
          </p>
        </details>
      </section>
      <div className="ws-toolbar ws-filters">
        <label>
          Queue
          <select value={state} onChange={(e) => setState(e.target.value)}>
            {[
              "pending",
              "flagged",
              "skipped",
              "reviewed",
              "excluded",
              "all",
            ].map((s) => (
              <option key={s} value={s}>
                {s === "pending"
                  ? "Awaiting review"
                  : s[0].toUpperCase() + s.slice(1)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Order
          <select value={order} onChange={(e) => setOrder(e.target.value)}>
            <option value="priority">Needs attention first</option>
            <option value="oldest">Oldest first</option>
            <option value="newest">Newest first</option>
          </select>
        </label>
        <label>
          Source
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="">All sources</option>
            {[
              "user-note",
              "world_teachings",
              "world_observations",
              "utterances",
              "observations",
            ].map((s) => (
              <option key={s} value={s}>
                {sourceName(s)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Search
          <input
            type="search"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Text or source ID"
          />
        </label>
      </div>
      <div className="ws-review">
        <section className="ws-card ws-queue">
          <div className="ws-toolbar">
            <h2>Review queue</h2>
            <span>{queue?.total ?? "…"} records</span>
          </div>
          {!!checked.length && (
            <div className="ws-toolbar">
              <button disabled={busy} onClick={() => decision("skip", checked)}>
                Skip {checked.length}
              </button>
              <button
                className="secondary"
                disabled={busy}
                onClick={() => decision("flag", checked)}
              >
                Flag {checked.length}
              </button>
            </div>
          )}
          <div className="ws-list">
            {queue?.items.map((row: Item) => (
              <div className="ws-queue-row" key={row.id}>
                <input
                  aria-label={"Select " + (row.preview || row.id)}
                  type="checkbox"
                  checked={checked.includes(row.id)}
                  onChange={(e) =>
                    setChecked(
                      e.target.checked
                        ? [...checked, row.id]
                        : checked.filter((id) => id !== row.id),
                    )
                  }
                />
                <button
                  className="ws-record"
                  aria-pressed={selected?.id === row.id}
                  disabled={busy}
                  onClick={() => open(row)}
                >
                  <strong>{row.preview || sourceName(row.source_kind)}</strong>
                  <span>{sourceName(row.source_kind)}</span>
                  <small>
                    {row.triage?.action === "flag" ? "Flagged · " : ""}
                    {row.annotation?.reviewed
                      ? "Human confirmed"
                      : row.annotation
                        ? "Suggested · unreviewed"
                        : "Unreviewed"}
                    {!row.allowed && " · Source excluded"}
                  </small>
                </button>
              </div>
            ))}
          </div>
          {queue?.total === 0 && (
            <Empty>
              No records match this queue. Skipped records are saved under
              Skipped. Clear the filters or{" "}
              <a href="/collection">collect a note or saved inputs</a>. If you
              have finished reviewing,{" "}
              <a href="/datasets">inspect eligible evidence in Datasets</a>.
            </Empty>
          )}
          <Pager
            offset={offset}
            total={queue?.total || 0}
            onChange={(n) => {
              setOffset(n);
              setChecked([]);
            }}
          />
        </section>
        <section className="ws-card ws-editor">
          {!selected || !raw ? (
            <Empty>
              Choose a record to begin. Suggestions are never confirmed
              automatically.
            </Empty>
          ) : (
            <>
              <div className="ws-toolbar">
                <h2>Review this memory</h2>
                <span className="ws-pill">
                  {selected.annotation?.reviewed
                    ? "Human reviewed"
                    : "Suggestion · not ground truth"}
                </span>
              </div>
              <div className="ws-evidence">
                <div className="eyebrow">Original evidence</div>
                <EvidenceMedia raw={raw} />
                <p>
                  {raw.payload.text ||
                    "Structured observation — inspect the source details below before writing a factual label."}
                </p>
                <small>
                  {sourceName(raw.source_kind)} · {raw.origin} ·{" "}
                  {date(raw.source_timestamp)}
                </small>
                <p>
                  <small>Source: {raw.source_id}</small>
                </p>
                <Details
                  value={raw}
                  title="Inspect original evidence and provenance"
                />
              </div>
              {["model", "derived"].includes(raw.origin) && (
                <p className="error">
                  This source is model-derived. Reviewing it will not make it
                  independent evidence for datasets.
                </p>
              )}
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  save();
                }}
                onChange={() => setDirty(true)}
              >
                <label>
                  Reviewed memory
                  <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    required={labelStatus !== "excluded"}
                  />
                </label>
                <label>
                  Which fact or relationship? (comma-separated)
                  <input
                    value={facts}
                    onChange={(e) => setFacts(e.target.value)}
                    placeholder="Example: toolbox:location"
                    required={labelStatus !== "excluded"}
                  />
                </label>
                <small>
                  Use the same name for competing claims about the same fact.
                  This helps detect contradictions and keeps related evidence in
                  one dataset split.
                </small>
                <label>
                  How should this memory be treated?
                  <select
                    value={labelStatus}
                    onChange={(e) => setLabelStatus(e.target.value)}
                  >
                    <option value="current">Current information</option>
                    <option value="historical">Historical information</option>
                    <option value="contested">Uncertain or conflicting</option>
                    <option value="excluded">
                      Irrelevant or unusable — exclude
                    </option>
                  </select>
                </label>
                <details>
                  <summary>Validity and review notes</summary>
                  <label>
                    Valid from (UTC)
                    <input
                      type="datetime-local"
                      step="0.001"
                      value={from}
                      onChange={(e) => setFrom(e.target.value)}
                    />
                  </label>
                  <label>
                    Valid until (UTC)
                    <input
                      type="datetime-local"
                      step="0.001"
                      value={until}
                      onChange={(e) => setUntil(e.target.value)}
                    />
                  </label>
                  <label>
                    Evidence quality / source reliability notes
                    <textarea
                      value={notes}
                      onChange={(e) => setNotes(e.target.value)}
                    />
                  </label>
                </details>
                <div className="ws-toolbar">
                  <button disabled={busy}>Save human review</button>
                  <button
                    type="button"
                    className="secondary"
                    disabled={busy}
                    onClick={() => decision("skip")}
                  >
                    Skip for later
                  </button>
                  <button
                    type="button"
                    className="secondary"
                    disabled={busy}
                    onClick={() => decision("flag")}
                  >
                    Flag problem
                  </button>
                </div>
                <small>
                  Corrections preserve history and invalidate datasets using the
                  older label. Review is human testimony, not a guarantee of
                  truth.
                </small>
              </form>
              <button
                className="secondary"
                disabled={busy}
                onClick={() => open(selected)}
              >
                Reload latest revision
              </button>
              <button
                className="secondary"
                disabled={busy}
                onClick={() =>
                  run(async () => {
                    await api("/policy", {
                      record_id: raw.id,
                      allowed: selected.allowed === false,
                      actor: "paired-reviewer",
                    });
                    setSelected(null);
                    setRaw(null);
                    setDirty(false);
                  }, "Source availability updated. Existing recall may need a new dataset.")
                }
              >
                {selected.allowed === false
                  ? "Allow this source again"
                  : "Exclude source from datasets & recall"}
              </button>
            </>
          )}
        </section>
      </div>
    </>
  );
}
