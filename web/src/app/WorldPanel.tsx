"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type { components } from "../api";

type World = components["schemas"]["WorldState"];
type Question = components["schemas"]["Question"];
type Reply = components["schemas"]["TeachingReply"];

async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch("/api/v2" + path, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

// Failed replies survive reloads with their original IDs and frozen proposition versions.
const outboxKey = "grounded-world-replies-v2";
function pending(): { question: string; reply: Reply }[] {
  try {
    return JSON.parse(localStorage.getItem(outboxKey) ?? "[]");
  } catch {
    return [];
  }
}

export default function WorldPanel({
  episodeId,
  enabled,
}: {
  episodeId?: string;
  enabled: boolean;
}) {
  const [world, setWorld] = useState<World | null>(null);
  const [error, setError] = useState("");
  const [answer, setAnswer] = useState("");
  const [expected, setExpected] = useState("");
  const [suggestedName, setSuggestedName] = useState("");
  const [cause, setCause] = useState("");
  const [horizon, setHorizon] = useState(30);
  const [selected, setSelected] = useState<Question | null>(null);
  const [choices, setChoices] = useState<{ id: string; label: string }[]>([]);
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState<string | null>(null);
  const [replacement, setReplacement] = useState("");
  const [reason, setReason] = useState("");
  const flushing = useRef<Promise<boolean> | null>(null);
  const lastTouch = useRef(0);
  const question = selected ?? world?.attention;
  const refresh = useCallback(async () => {
    setWorld(
      await request<World>(
        "/world" +
          (episodeId ? "?episode_id=" + encodeURIComponent(episodeId) : ""),
      ),
    );
  }, [episodeId]);
  const flush = useCallback(async (): Promise<boolean> => {
    if (flushing.current) return flushing.current;
    const work = async () => {
      let clarified = true;
      for (const item of pending()) {
        const result = await request<{
          needs_clarification?: boolean;
          matches?: { id: string; label: string }[];
        }>("/questions/" + item.question + "/answers", item.reply);
        if (result.needs_clarification) {
          clarified = false;
          const snapshot = await request<World>("/world");
          setSelected(
            snapshot.questions.find((q) => q.id === item.question) ?? null,
          );
          setChoices(result.matches ?? []);
          setAnswer(item.reply.text);
          setError(
            "Choose whether this is a remembered thing or a new one, then send your answer.",
          );
        }
        localStorage.setItem(
          outboxKey,
          JSON.stringify(pending().filter((p) => p.reply.id !== item.reply.id)),
        );
      }
      return clarified;
    };
    flushing.current = work();
    try {
      return await flushing.current;
    } finally {
      flushing.current = null;
    }
  }, []);
  useEffect(() => {
    if (!enabled) return;
    const update = async () => {
      try {
        await flush();
        await refresh();
      } catch (e) {
        setError(
          "Live inference unavailable. Saved answers will retry. " + String(e),
        );
      }
    };
    void update();
    const timer = setInterval(update, 2000);
    return () => clearInterval(timer);
  }, [enabled, flush, refresh]);
  useEffect(() => {
    if (!selected) {
      setAnswer("");
      setExpected("");
      setChoices([]);
      setSuggestedName("");
      setCause("");
    }
  }, [question?.id]);
  if (!enabled || !world) return null;

  async function send(response: Reply["response"], entityId?: string) {
    if (!question) return;
    setBusy(true);
    setError("");
    const reply: Reply = {
      id: crypto.randomUUID(),
      question_version: question.version,
      text: answer,
      response,
      teacher_id: "local-human",
      label: suggestedName || null,
      cause_event: cause || null,
      entity_id: entityId ?? null,
      expected_event: expected || null,
      horizon_seconds: horizon,
    };
    localStorage.setItem(
      outboxKey,
      JSON.stringify([...pending(), { question: question.id, reply }]),
    );
    try {
      const complete = await flush();
      await refresh();
      if (complete) {
        setSelected(null);
        setAnswer("");
        setExpected("");
        setChoices([]);
        setSuggestedName("");
        setCause("");
      }
    } catch (e) {
      setError("Answer saved on this browser for retry. " + String(e));
    } finally {
      setBusy(false);
    }
  }
  async function interpret() {
    if (!question) return;
    setBusy(true);
    try {
      const result = await request<{
        label: string;
        cause_event: string;
        expected_event: string;
      }>(`/questions/${question.id}/interpret`, { text: answer });
      setSuggestedName(result.label);
      setCause(result.cause_event);
      setExpected(result.expected_event);
    } catch (e) {
      setError(
        "You can save your original answer without interpretation. " +
          String(e),
      );
    } finally {
      setBusy(false);
    }
  }
  async function playback(element: HTMLMediaElement) {
    if (!episodeId || !world) return;
    element.pause();
    try {
      const duration = Number.isFinite(element.duration)
        ? Math.min(240, element.duration)
        : 30;
      await request("/output", { duration_seconds: duration });
      element.onplay = null;
      await element.play();
    } catch (e) {
      setError(
        "Playback paused because exclusion could not be recorded. " + String(e),
      );
    }
  }
  async function revise(belief: World["beliefs"][number], operation: string) {
    setBusy(true);
    try {
      await request(`/beliefs/${belief.id}/revisions`, {
        id: crypto.randomUUID(),
        expected_version: belief.version,
        operation,
        reason: reason || "Corrected from the review console",
        value: operation === "correct" ? replacement : null,
      });
      setEditing(null);
      setReplacement("");
      setReason("");
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="card world-panel" aria-label="World model">
      <div className="row">
        <h2>My understanding of the world</h2>
        <span className="badge">{world.status}</span>
      </div>
      <p>
        <strong>{world.location.label}</strong> · {world.location.certainty}
        {world.location.certainty === "unknown" &&
          " — I can keep observing while I work out where I am."}
      </p>
      <p className="muted">
        <span className="badge">{world.executive.mode}</span>{" "}
        {world.executive.reason}
        {world.executive.request && <> · Next: {world.executive.request}</>}
      </p>
      {world.executive.working_set.length > 0 && (
        <details>
          <summary>Working set ({world.executive.working_set.length})</summary>
          <ul>
            {world.executive.working_set.map((item) => (
              <li key={item.kind + item.id}>
                {item.kind}: {item.label} <small className="muted">({item.basis})</small>
              </li>
            ))}
          </ul>
        </details>
      )}
      {world.location.candidates.length > 0 && (
        <small className="muted">
          Possible places:{" "}
          {world.location.candidates.map((c) => c.label).join(", ")}
        </small>
      )}
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      {question ? (
        <div className="world-question">
          <span className="eyebrow">
            {question.kind} · {question.reason}
          </span>
          <h3>{question.text}</h3>
          {question.region?.asset && (
            <img
              className="crop"
              src={"/api/assets/" + question.region.asset}
              alt="Frozen evidence for this question"
            />
          )}
          {!question.region &&
            question.evidence_refs
              .filter((ref) => /\.(jpg|png)$/.test(ref))
              .map((ref) => (
                <img
                  className="crop"
                  key={ref}
                  src={"/api/assets/" + ref}
                  alt="Observed scene"
                />
              ))}
          {question.evidence_refs
            .filter((ref) => /\.wav$/.test(ref))
            .map((ref) => (
              <audio
                key={ref}
                controls
                src={"/api/assets/" + ref}
                onPlay={(e) => {
                  if (!e.currentTarget.dataset.excluded) {
                    e.currentTarget.dataset.excluded = "yes";
                    void playback(e.currentTarget);
                  }
                }}
                onEnded={(e) => {
                  delete e.currentTarget.dataset.excluded;
                }}
              />
            ))}
          <label>
            Your answer
            <textarea
              value={answer}
              onChange={(e) => {
                setAnswer(e.target.value);
                if (Date.now() - lastTouch.current > 5000) {
                  lastTouch.current = Date.now();
                  void request(`/questions/${question.id}/touch`, {}).catch(
                    () => {},
                  );
                }
              }}
              placeholder={
                question.kind === "explain"
                  ? "What happened, and why?"
                  : "Use your own words"
              }
            />
          </label>
          <button
            className="secondary"
            disabled={busy || !answer.trim()}
            onClick={interpret}
          >
            Help interpret my answer locally
          </button>
          {suggestedName && (
            <label>
              Proposed name (review before saving)
              <input
                value={suggestedName}
                onChange={(e) => setSuggestedName(e.target.value)}
              />
            </label>
          )}
          {["explain", "clarify", "predict"].includes(question.kind) && (
            <details>
              <summary>Teach an expectation</summary>
              <label>
                What triggers it? (optional)
                <input
                  value={cause}
                  onChange={(e) => setCause(e.target.value)}
                />
              </label>
              <label>
                What should happen next?
                <input
                  value={expected}
                  onChange={(e) => setExpected(e.target.value)}
                  placeholder="A sound or event you have shown me"
                />
              </label>
              <label>
                Within how many seconds?
                <input
                  type="number"
                  min={1}
                  max={3600}
                  value={horizon}
                  onChange={(e) => setHorizon(Number(e.target.value))}
                />
              </label>
            </details>
          )}
          {choices.length > 0 && (
            <div className="row">
              {choices.map((choice) => (
                <button
                  key={choice.id}
                  onClick={() => send("answer", choice.id)}
                >
                  Same {choice.label}
                </button>
              ))}
              <button onClick={() => send("answer", "new")}>
                New, with the same name
              </button>
            </div>
          )}
          <div className="row">
            {question.kind === "intervene" && (
              <>
                <button disabled={busy} onClick={() => send("yes")}>
                  I&apos;ll do it now
                </button>
                <button disabled={busy} onClick={() => send("no")}>
                  Not now
                </button>
              </>
            )}
            {question.proposition.entity_id && (
              <>
                <button disabled={busy} onClick={() => send("yes")}>
                  Yes
                </button>
                <button disabled={busy} onClick={() => send("no")}>
                  No
                </button>
              </>
            )}
            <button
              disabled={busy || !answer.trim()}
              onClick={() => send("answer")}
            >
              Remember my answer
            </button>
            <button
              disabled={busy}
              className="secondary"
              onClick={() => send("unknown")}
            >
              I don’t know
            </button>
            <button
              disabled={busy}
              className="secondary"
              onClick={() => send("later")}
            >
              Later
            </button>
            <button
              disabled={busy}
              className="secondary"
              onClick={() => send("dismiss")}
            >
              Not important
            </button>
          </div>
        </div>
      ) : (
        <p className="muted">
          Observing and learning. I’ll ask when a question would help.
        </p>
      )}
      <details>
        <summary>
          Questions saved for later (
          {world.questions.filter((q) => q.status === "open").length})
        </summary>
        {world.questions
          .filter((q) => q.status === "open")
          .map((q) => (
            <button
              className="session"
              key={q.id}
              onClick={() => setSelected(q)}
            >
              {q.text}
            </button>
          ))}
      </details>
      <details open={world.predictions.some((p) => p.status === "pending")}>
        <summary>What I expect</summary>
        {world.predictions
          .slice(-8)
          .reverse()
          .map((p) => (
            <p key={p.id}>
              <strong>{p.label}</strong> · within {Math.round(p.due - p.start)}{" "}
              seconds · {p.status.replaceAll("_", " ")}
            </p>
          ))}
        {!world.predictions.length && (
          <p className="muted">
            Expectations grow from repeated experience and your explanations.
          </p>
        )}
      </details>
      <details>
        <summary>Recent observations and surprises</summary>
        {world.recent_events
          .slice(-12)
          .reverse()
          .map((e) => (
            <p key={e.id}>
              {e.surprising ? "Unexpected: " : "Observed: "}
              {e.label}
              {e.kind === "sound" ? " · source location uncertain" : ""}
            </p>
          ))}
      </details>
      <details>
        <summary>Beliefs and their evidence ({world.beliefs.length})</summary>
        {world.beliefs
          .slice()
          .reverse()
          .map((b) => (
            <article key={b.id} className="memory">
              <strong>{b.subject_label}</strong> · {b.predicate}
              <p>{b.value}</p>
              <small>
                {b.status} · {b.certainty} · source: {b.source.kind}
                {b.context_label ? " · " + b.context_label : ""} ·{" "}
                {b.supporting.length} evidence links / {b.conflicting.length}{" "}
                conflicts
              </small>
              <details>
                <summary>Evidence and revision history</summary>
                <p className="muted">
                  {b.supporting.join(", ") || "Built-in revisable prior"}
                </p>
                {world.revisions
                  .filter((r) => r.belief_id === b.id)
                  .map((r) => (
                    <p key={r.id}>
                      {r.reason}: {r.before.value} → {r.after.value} (
                      {r.after.status})
                    </p>
                  ))}
              </details>
              {editing === b.id ? (
                <>
                  <label>
                    Replacement statement
                    <input
                      value={replacement}
                      onChange={(e) => setReplacement(e.target.value)}
                    />
                  </label>
                  <label>
                    Why change this?
                    <input
                      value={reason}
                      onChange={(e) => setReason(e.target.value)}
                    />
                  </label>
                  <div className="row">
                    <button
                      disabled={busy || !replacement.trim()}
                      onClick={() => revise(b, "correct")}
                    >
                      Correct
                    </button>
                    <button
                      disabled={busy}
                      onClick={() => revise(b, "contradict")}
                    >
                      I disagree
                    </button>
                    <button
                      disabled={busy}
                      onClick={() => revise(b, "retract")}
                    >
                      Retract
                    </button>
                    <button onClick={() => setEditing(null)}>Cancel</button>
                  </div>
                </>
              ) : (
                <button
                  className="secondary"
                  onClick={() => {
                    setEditing(b.id);
                    setReplacement(b.value);
                  }}
                >
                  Revise
                </button>
              )}
            </article>
          ))}
      </details>
    </section>
  );
}
