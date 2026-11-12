"use client";
import { useState } from "react";
import {
  api,
  ArtifactHistory,
  Details,
  Empty,
  Item,
  Metrics,
  Pager,
  short,
  sourceName,
  useLoad,
  useWork,
  useArtifactOptions,
} from "./shared";
import { artifactLink, readinessAdvice } from "./guideModel";
const splitName: Record<string, string> = {
  train: "Training / development",
  validation: "Validation / tuning",
  test: "Final evaluation · held out",
  regression: "Regression · held out",
};
const categories = [
  "accuracy",
  "retrieval",
  "temporal",
  "contradiction",
  "consolidation",
  "grounding",
  "privacy",
];
export function Datasets() {
  const { run, busy } = useWork();
  const [offset, setOffset] = useState(0),
    [source, setSource] = useState(""),
    [selected, setSelected] = useState<Record<string, Item>>({});
  const eligible = useLoad(
    `/records?state=eligible&offset=${offset}&source=${source}`,
  );
  const [groups, setGroups] = useState<Item[] | null>(null),
    [name, setName] = useState("Reviewed memories"),
    [cases, setCases] = useState<Item[]>([]),
    [result, setResult] = useState<Item | null>(null);
  const [query, setQuery] = useState(""),
    [split, setSplit] = useState("validation"),
    [expected, setExpected] = useState<string[]>([]),
    [category, setCategory] = useState<string[]>(["accuracy", "retrieval"]),
    [uncertain, setUncertain] = useState(false),
    [caseGroup, setCaseGroup] = useState(""),
    [asOf, setAsOf] = useState(""),
    [history, setHistory] = useState(false),
    [hitCount, setHitCount] = useState(""),
    [restrictSources, setRestrictSources] = useState(false),
    [allowedSources, setAllowedSources] = useState<string[]>([]);
  const [evaluationTime, setEvaluationTime] = useState(() => Date.now() / 1000);
  const assignments: Record<string, string> = Object.fromEntries(
    (groups || []).flatMap((g) =>
      g.record_ids.map((id: string) => [id, g.split]),
    ),
  );
  function clearPlan() {
    setExpected([]);
    setGroups(null);
    setCases([]);
    setResult(null);
    setEvaluationTime(Date.now() / 1000);
  }
  return (
    <>
      <section className="ws-next">
        <h2>Build from reviewed evidence</h2>
        <p>
          Select records, keep related evidence together, then write independent
          questions and expected answers. New groups start in validation;
          held-out groups retain their existing split.
        </p>
        <small>
          Automatic grouping checks shared sources, conversations, facts, and
          exact duplicates. Undeclared relationships and paraphrases still
          require your judgment.
        </small>
      </section>
      <section className="ws-card">
        <h2>1. Select eligible records</h2>
        <div className="ws-toolbar">
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
          <span>
            {Object.keys(selected).length} selected · {eligible?.total || 0}{" "}
            match this filter
          </span>
          <a href="/labeling?view=all">
            Inspect unreviewed or excluded records →
          </a>
        </div>
        <div className="ws-list">
          {eligible?.items.map((r: Item) => (
            <label key={r.id} className="ws-check">
              <input
                type="checkbox"
                checked={!!selected[r.id]}
                onChange={(e) => {
                  const next = { ...selected };
                  if (e.target.checked) next[r.id] = r;
                  else delete next[r.id];
                  setSelected(next);
                  clearPlan();
                }}
              />
              <span>
                <strong>{r.preview}</strong>
                <small>
                  {r.annotation.status} · {sourceName(r.source_kind)}
                </small>
              </span>
            </label>
          ))}
        </div>
        {eligible?.total === 0 && (
          <Empty>
            Review a record in Labeling first. Excluded, unreviewed, and
            model-derived inputs cannot enter a dataset.{" "}
            <a href="/labeling">Open Labeling →</a>
          </Empty>
        )}
        <Pager
          offset={offset}
          total={eligible?.total || 0}
          onChange={setOffset}
        />
        <button
          disabled={busy || !Object.keys(selected).length}
          onClick={() =>
            run(async () => {
              const plan = await api("/dataset-plan", {
                record_ids: Object.keys(selected),
              });
              setGroups(plan.groups);
              setCases([]);
            })
          }
        >
          Group selected evidence
        </button>
        {!Object.keys(selected).length && (
          <small className="gm-reason">
            Select at least one eligible record to group evidence.
          </small>
        )}
      </section>
      {groups && (
        <>
          <section className="ws-card">
            <h2>2. Choose splits</h2>
            <p>
              Use validation to compare candidates. Keep final evaluation and
              regression groups out of tuning. Once prepared, split ownership is
              permanent in this ledger.
            </p>
            {groups.map((g, i) => (
              <div className="ws-split" key={g.record_ids.join()}>
                <div>
                  {g.record_ids.map((id: string) => (
                    <p key={id}>{selected[id]?.preview || short(id)}</p>
                  ))}
                </div>
                <label>
                  Split {g.locked && "(already reserved)"}
                  <select
                    disabled={g.locked || !!cases.length}
                    value={g.split}
                    onChange={(e) =>
                      setGroups(
                        groups.map((v, j) =>
                          i === j ? { ...v, split: e.target.value } : v,
                        ),
                      )
                    }
                  >
                    {Object.entries(splitName).map(([v, label]) => (
                      <option key={v} value={v}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            ))}
          </section>
          <section className="ws-card">
            <h2>3. Add independently reviewed questions</h2>
            <p>
              Do not copy a suggested label into a question and assume it
              measures accuracy. Write what a person would actually ask and
              select the evidence that should answer it.
            </p>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                setCases([
                  ...cases,
                  {
                    id: crypto.randomUUID(),
                    query,
                    split,
                    expected_ids: expected,
                    categories: category,
                    expected_uncertain: uncertain,
                    expected_hit_count:
                      hitCount === "" ? null : Number(hitCount),
                    as_of: asOf ? Date.parse(asOf + "Z") / 1000 : null,
                    include_history: history,
                    allowed_sources: restrictSources ? allowedSources : null,
                    group_ids: caseGroup.trim()
                      ? [caseGroup.trim()]
                      : [
                          ...new Set(
                            expected.flatMap((id) => selected[id].group_ids),
                          ),
                        ],
                  },
                ]);
                setQuery("");
                setExpected([]);
                setCaseGroup("");
              }}
            >
              <div className="ws-columns">
                <label>
                  Question
                  <input
                    required
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                </label>
                <label>
                  Question split
                  <select
                    value={split}
                    onChange={(e) => {
                      setSplit(e.target.value);
                      setExpected([]);
                    }}
                  >
                    {["validation", "test", "regression"].map((s) => (
                      <option key={s} value={s}>
                        {splitName[s]}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <p>
                Expected evidence (leave all unchecked for a question that
                cannot be answered):
              </p>
              {Object.keys(assignments)
                .filter((id) => assignments[id] === split)
                .map((id) => (
                  <label key={id} className="ws-check">
                    <input
                      type="checkbox"
                      checked={expected.includes(id)}
                      onChange={(e) =>
                        setExpected(
                          e.target.checked
                            ? [...expected, id]
                            : expected.filter((v) => v !== id),
                        )
                      }
                    />
                    <span>{selected[id]?.preview || id}</span>
                  </label>
                ))}
              <label>
                Question source / conversation{" "}
                {expected.length
                  ? "(optional)"
                  : "(required for an unanswerable question)"}
                <input
                  value={caseGroup}
                  required={!expected.length}
                  onChange={(e) => setCaseGroup(e.target.value)}
                  placeholder="Identify related question sources consistently across datasets"
                />
              </label>
              <div className="ws-toolbar">
                {categories.map((c) => (
                  <label key={c} className="ws-check">
                    <input
                      type="checkbox"
                      checked={category.includes(c)}
                      onChange={(e) =>
                        setCategory(
                          e.target.checked
                            ? [...category, c]
                            : category.filter((v) => v !== c),
                        )
                      }
                    />
                    {c}
                  </label>
                ))}
              </div>
              <label className="ws-check">
                <input
                  type="checkbox"
                  checked={uncertain}
                  onChange={(e) => setUncertain(e.target.checked)}
                />
                The answer should report uncertainty or conflicting evidence
              </label>
              <details>
                <summary>
                  Historical, duplicate, and source-access checks
                </summary>
                <label className="ws-check">
                  <input
                    type="checkbox"
                    checked={restrictSources}
                    onChange={(e) => setRestrictSources(e.target.checked)}
                  />
                  Restrict sources available to this question
                </label>
                {restrictSources && (
                  <>
                    <p>
                      Select allowed sources. Selecting none tests an
                      access-denied answer.
                    </p>
                    {[
                      ...new Set(
                        Object.values(selected).map((r) => r.source_id),
                      ),
                    ].map((id) => (
                      <label className="ws-check" key={id}>
                        <input
                          type="checkbox"
                          checked={allowedSources.includes(id)}
                          onChange={(e) =>
                            setAllowedSources(
                              e.target.checked
                                ? [...allowedSources, id]
                                : allowedSources.filter((v) => v !== id),
                            )
                          }
                        />
                        {id}
                      </label>
                    ))}
                  </>
                )}
                <label>
                  Ask as of (UTC)
                  <input
                    type="datetime-local"
                    value={asOf}
                    onChange={(e) => setAsOf(e.target.value)}
                  />
                </label>
                <label className="ws-check">
                  <input
                    type="checkbox"
                    checked={history}
                    onChange={(e) => setHistory(e.target.checked)}
                  />
                  Include historical memories
                </label>
                <label>
                  Expected number of consolidated answers (optional)
                  <input
                    type="number"
                    min="0"
                    value={hitCount}
                    onChange={(e) => setHitCount(e.target.value)}
                  />
                </label>
              </details>
              <button
                disabled={
                  !query.trim() ||
                  !category.length ||
                  !Object.values(assignments).includes(split)
                }
              >
                Add question
              </button>
              {(!query.trim() ||
                !category.length ||
                (!expected.length && !caseGroup.trim())) && (
                <small className="gm-reason">
                  Write a question, choose at least one category, and select its
                  expected evidence. For an unanswerable question, leave
                  evidence unchecked and supply its source or conversation.
                </small>
              )}
            </form>
            {cases.map((c, i) => (
              <div className="ws-inspect" key={c.id}>
                <strong>{c.query}</strong>
                <p>
                  {splitName[c.split]} · {c.expected_ids.length} expected
                  records · {c.categories.join(", ")}
                </p>
                <button
                  className="secondary"
                  onClick={() => setCases(cases.filter((_, j) => j !== i))}
                >
                  Remove question
                </button>
              </div>
            ))}
          </section>
          <section className="ws-card">
            <h2>4. Prepare a version</h2>
            <label>
              Dataset name
              <input value={name} onChange={(e) => setName(e.target.value)} />
            </label>
            <p>
              {Object.keys(assignments).length} records · {cases.length}{" "}
              reviewed questions. Source and label revisions will be pinned.
            </p>
            <button
              disabled={busy || !cases.length || !name.trim()}
              onClick={() =>
                run(
                  async () =>
                    setResult(
                      await api("/datasets", {
                        name,
                        assignments,
                        cases,
                        evaluation_time: evaluationTime,
                        reviewer: "paired-reviewer",
                        purpose: "reviewed-local",
                      }),
                    ),
                  "Dataset prepared. You can now freeze an experiment.",
                )
              }
            >
              Prepare dataset
            </button>
            {(!cases.length || !name.trim()) && (
              <small className="gm-reason">
                Add at least one independent reviewed question and a dataset
                name before preparing.
              </small>
            )}
            {result && (
              <>
                <h3>{result.spec.name} is ready</h3>
                <p>
                  {result.records.length} records · {result.warnings.length}{" "}
                  duplicate or conflict warnings
                </p>
                {result.warnings.map((w: Item, i: number) => (
                  <p key={i}>
                    {w.kind.replaceAll("_", " ")}:{" "}
                    {w.record_ids.map(short).join(", ")}
                  </p>
                ))}
                <a href={artifactLink("experiments", result.id)}>
                  Use this dataset in Experiments →
                </a>
              </>
            )}
          </section>
        </>
      )}
      <details className="gm-disclosure">
        <summary>Dataset versions and composition</summary>
        <ArtifactHistory kind="dataset" />
      </details>
    </>
  );
}
export function Experiments() {
  const { run, busy } = useWork();
  const {
    options: datasets,
    selected: dataset,
    setSelected: setDataset,
    linkError,
    loading,
  } = useArtifactOptions("dataset", "dataset");
  const [algorithm, setAlgorithm] = useState("coverage"),
    [top, setTop] = useState(3),
    [score, setScore] = useState(0.5),
    [result, setResult] = useState<Item | null>(null);
  return (
    <>
      <section className="ws-next">
        <h2>Compare retrieval, without training a model</h2>
        <p>
          These experiments freeze a deterministic search configuration. They do
          not change model weights, perception, or world beliefs.
        </p>
      </section>
      {linkError && (
        <p role="alert" className="error">
          {linkError} <a href="/datasets">Open Datasets →</a>
        </p>
      )}
      {!loading && !datasets.length && (
        <Empty>
          No prepared datasets yet.{" "}
          <a href="/datasets">
            Prepare reviewed evidence and questions first →
          </a>
        </Empty>
      )}
      <div className="ws-columns">
        <form
          className="ws-card"
          onSubmit={(e) => {
            e.preventDefault();
            run(async () => {
              const experiment = await api("/experiments", {
                dataset_id: dataset,
                config: { algorithm, top_k: top, min_score: score },
              });
              setResult(experiment);
            }, "Candidate frozen. Run validation before final evaluation.");
          }}
        >
          <h2>New candidate</h2>
          <p>
            A candidate saves search settings for one fixed dataset. It stays
            inactive until evaluated and explicitly approved.
          </p>
          <label>
            Dataset
            <select
              value={dataset}
              disabled={busy}
              required
              onChange={(e) => setDataset(e.target.value)}
            >
              <option value="">Choose a prepared version</option>
              {datasets?.map((d: Item) => (
                <option key={d.id} value={d.id}>
                  {d.name} · {d.records} records · {d.id.slice(-6)}
                </option>
              ))}
            </select>
          </label>
          <small>
            A dataset is a fixed version of reviewed evidence and independent
            questions.
          </small>
          <label>
            Retrieval strategy
            <select
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value)}
            >
              <option value="coverage">Query coverage</option>
              <option value="overlap">Word overlap · fixed baseline</option>
            </select>
          </label>
          <p className="muted">
            {algorithm === "coverage"
              ? "Rewards memories matching more of the question’s words."
              : "Compares shared words with all words in the question and memory."}
          </p>
          <details>
            <summary>Optional search limits</summary>
            <label>
              Maximum answers
              <input
                type="number"
                min="1"
                max="20"
                required
                value={top}
                onChange={(e) => setTop(Number(e.target.value))}
              />
            </label>
            <label>
              Minimum match score
              <input
                type="number"
                min="0.01"
                max="1"
                step="0.01"
                required
                value={score}
                onChange={(e) => setScore(Number(e.target.value))}
              />
            </label>
            <small>
              The score is word matching, not calibrated confidence.
            </small>
          </details>
          <p>
            <button disabled={busy || !dataset}>Freeze candidate</button>
          </p>
          {!dataset && (
            <small className="gm-reason">
              Choose a prepared dataset to freeze a candidate.
            </small>
          )}
          {result && (
            <>
              <p>Candidate {short(result.id)} saved.</p>
              <a
                href={"/evaluation?candidate=" + encodeURIComponent(result.id)}
              >
                Validate this candidate →
              </a>
            </>
          )}
        </form>
        <details className="gm-disclosure">
          <summary>How to compare settings and local models</summary>
          <p>
            Use validation evidence to compare relevant matches, missed
            evidence, and fully correct answers. Change one setting at a time.
          </p>
          <h3>Local models</h3>
          <p>
            Existing DINO, SAM, Whisper, and optional local language
            interpretation remain in capture and world learning. They are not
            retrieval training jobs.
          </p>
          <p>
            Embeddings and reranking need a reviewed benchmark before adding
            another model or service. No weights are downloaded here.
          </p>
          <p className="error">
            Freeze your choice before opening final evaluation results. Never
            tune to those questions.
          </p>
        </details>
      </div>
      <details className="gm-disclosure">
        <summary>Candidate history</summary>
        <ArtifactHistory kind="experiment" />
      </details>
    </>
  );
}
export function Evaluation() {
  const { run, busy } = useWork();
  const {
    options: experiments,
    selected: candidate,
    setSelected: setCandidate,
    linkError,
    loading,
  } = useArtifactOptions("experiment", "candidate");
  const [split, setSplit] = useState("validation"),
    [frozen, setFrozen] = useState(false),
    [result, setResult] = useState<Item | null>(null);
  return (
    <>
      <section className="ws-next">
        <h2>Understand the evidence behind improvement</h2>
        <p>
          Relevant matches measure precision. Expected evidence found measures
          recall. A fully correct case also checks uncertainty and duplicate
          consolidation.
        </p>
        <small>
          Results describe the reviewed questions you supplied, not general
          truth or real-world recognition accuracy.
        </small>
      </section>
      {linkError && (
        <p role="alert" className="error">
          {linkError} <a href="/experiments">Open Experiments →</a>
        </p>
      )}
      {!loading && !experiments.length && (
        <Empty>
          No candidates yet.{" "}
          <a href="/experiments">Freeze a search candidate first →</a>
        </Empty>
      )}
      <form
        className="ws-card"
        onSubmit={(e) => {
          e.preventDefault();
          run(
            async () =>
              setResult(
                await api("/evaluations", { experiment_id: candidate, split }),
              ),
            "Evaluation saved with per-question results and a fixed baseline.",
          );
        }}
      >
        <h2>Run a comparison</h2>
        <div className="ws-columns">
          <label>
            Frozen candidate
            <select
              required
              value={candidate}
              disabled={busy}
              onChange={(e) => {
                setCandidate(e.target.value);
                setFrozen(false);
              }}
            >
              <option value="">Choose a candidate</option>
              {experiments?.map((c: Item) => (
                <option key={c.id} value={c.id}>
                  {c.config.algorithm} · score {c.config.min_score} ·{" "}
                  {c.id.slice(-8)}
                </option>
              ))}
            </select>
          </label>
          <label>
            Evidence split
            <select
              value={split}
              disabled={busy}
              onChange={(e) => {
                setSplit(e.target.value);
                setFrozen(false);
              }}
            >
              {["validation", "test", "regression"].map((s) => (
                <option key={s} value={s}>
                  {splitName[s]}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p className="gm-reason">
          Validation helps choose settings. Final evaluation and regression use
          held-out questions that must stay out of tuning.
        </p>
        {split !== "validation" && (
          <label className="ws-check">
            <input
              type="checkbox"
              required
              checked={frozen}
              onChange={(e) => setFrozen(e.target.checked)}
            />
            <span>
              I have frozen this candidate. I will not use these final results
              to tune it.
            </span>
          </label>
        )}
        <button
          disabled={busy || !candidate || (split !== "validation" && !frozen)}
        >
          Run evaluation
        </button>
        {!candidate && (
          <small className="gm-reason">Choose a candidate to evaluate.</small>
        )}
        {candidate && split !== "validation" && !frozen && (
          <small className="gm-reason">
            Confirm that this is your frozen choice before running held-out
            questions.
          </small>
        )}
      </form>
      {result && <EvaluationResult value={result} />}
      <details className="gm-disclosure">
        <summary>Comparison history</summary>
        <p>
          Open a run to compare metrics and inspect each case. Final results
          remain separate from validation.
        </p>
        <ArtifactHistory kind="evaluation" onSelect={setResult} />
      </details>
    </>
  );
}
function EvaluationResult({ value }: { value: Item }) {
  const { run, busy } = useWork();
  return (
    <section className="ws-card">
      <h2>{splitName[value.split]} results</h2>
      <p>
        Candidate {short(value.experiment_id)} · dataset{" "}
        {short(value.dataset_id)}
      </p>
      <p>
        <a href={artifactLink("deployment", value.experiment_id)}>
          Check this candidate’s deployment readiness →
        </a>{" "}
        ·{" "}
        <a href={artifactLink("evaluation", value.experiment_id)}>
          Run another split for this candidate →
        </a>
      </p>
      <p className="gm-reason">
        This run is saved. Readiness separately checks both final suites, every
        required category, and the current evidence.
      </p>
      <Metrics value={value.candidate.metrics} />
      <p>
        Fixed baseline: {Math.round(value.baseline.metrics.precision * 100)}%
        relevant · {Math.round(value.baseline.metrics.recall * 100)}% found ·{" "}
        {Math.round(value.baseline.metrics.exact * 100)}% fully correct
      </p>
      {["precision", "recall", "exact"].some(
        (k) => value.candidate.metrics[k] < value.baseline.metrics[k],
      ) && (
        <p className="error">Regression detected against the fixed baseline.</p>
      )}
      <div className="ws-toolbar">
        {Object.entries(value.candidate.metrics.by_category).map(
          ([key, val]) => (
            <span className="ws-pill" key={key}>
              {key}: {Math.round(Number(val) * 100)}%
            </span>
          ),
        )}
      </div>
      {value.deployment_candidate && (
        <p>
          Against the full deployment corpus:{" "}
          {Math.round(value.deployment_candidate.metrics.exact * 100)}% fully
          correct. Promotion checks both views.
        </p>
      )}
      {value.candidate.cases.map((c: Item) => {
        const missing = c.expected_ids.filter(
            (id: string) => !c.retrieved_ids.includes(id),
          ),
          extra = c.retrieved_ids.filter(
            (id: string) => !c.expected_ids.includes(id),
          );
        const ids = [
          ...new Set<string>([
            ...missing,
            ...extra,
            ...(!c.exact ? c.expected_ids : []),
          ]),
        ];
        return (
          <div key={c.case_id} className="ws-inspect">
            <strong>
              {c.exact ? "✓ Passed" : "Needs attention"} ·{" "}
              {c.query || short(c.case_id)}
            </strong>
            <p>
              Missed evidence: {missing.length} · Incorrect matches:{" "}
              {extra.length} ·{" "}
              {c.response.uncertain
                ? "Uncertain / conflicting answer"
                : "No reported conflict"}
            </p>
            {c.response.hits.map((hit: Item, i: number) => (
              <p key={i}>
                {hit.text} <small>({hit.status})</small>
              </p>
            ))}
            {!!ids.length && (
              <div className="ws-toolbar">
                <button
                  className="secondary"
                  disabled={busy}
                  onClick={() =>
                    run(
                      () =>
                        api("/triage", {
                          record_ids: ids,
                          action: "flag",
                          actor: "evaluation-reviewer",
                          notes: `Inspect evaluation ${value.id}, case ${c.case_id}. Preserve held-out split ownership.`,
                        }),
                      "Evidence flagged for review. Existing labels and split ownership are unchanged.",
                    )
                  }
                >
                  Send evidence to Labeling
                </button>
                <a href={"/labeling?record=" + encodeURIComponent(ids[0])}>
                  Inspect first record →
                </a>
              </div>
            )}
            <Details
              value={c}
              title="Expected evidence and returned citations"
            />
          </div>
        );
      })}
    </section>
  );
}
export function Deployment() {
  const { run, busy } = useWork();
  const status = useLoad("/status"),
    health = useLoad("/health");
  const {
    options: experiments,
    selected: candidate,
    setSelected: setCandidate,
    linkError,
    loading,
  } = useArtifactOptions("experiment", "candidate");
  const [ready, setReady] = useState<Item | null>(null),
    [approved, setApproved] = useState(false),
    [target, setTarget] = useState<Item | null>(null),
    [rollbackApproved, setRollbackApproved] = useState(false),
    [query, setQuery] = useState(""),
    [answer, setAnswer] = useState<Item | null>(null);
  return (
    <>
      <section className="ws-next">
        <div className="eyebrow">Deployment scope</div>
        <h2>Reviewed-memory recall only</h2>
        <p>
          Promotion changes the dataset and search configuration used for
          reviewed recall. Live sensing, pretrained perception, and the world
          learner keep their existing behavior.
        </p>
        <span className="ws-pill">
          {health?.reason || "Checking deployment health…"}
        </span>
      </section>
      {linkError && (
        <p role="alert" className="error">
          {linkError} <a href="/experiments">Open Experiments →</a>
        </p>
      )}
      {!loading && !experiments.length && (
        <Empty>
          No candidates yet.{" "}
          <a href="/experiments">
            Create a candidate before checking readiness →
          </a>
        </Empty>
      )}
      <div className="ws-columns">
        <section className="ws-card">
          <h2>Currently active</h2>
          {status?.deployed ? (
            <>
              <p>
                {status.deployed.config.algorithm} · maximum{" "}
                {status.deployed.config.top_k} answers · minimum score{" "}
                {status.deployed.config.min_score}
              </p>
              <p>Dataset: {short(status.deployed.dataset_id)}</p>
              {status.deployed.synthetic && (
                <p className="error">
                  Synthetic regression deployment — not a real-world accuracy
                  result.
                </p>
              )}
              <Details value={status.deployed} />
            </>
          ) : (
            <Empty>No reviewed-recall configuration has been deployed.</Empty>
          )}
        </section>
        <section className="ws-card">
          <h2>Promotion readiness</h2>
          <label>
            Candidate
            <select
              value={candidate}
              disabled={busy}
              onChange={(e) => {
                setCandidate(e.target.value);
                setReady(null);
                setApproved(false);
              }}
            >
              <option value="">Choose an evaluated candidate</option>
              {experiments?.map((c: Item) => (
                <option key={c.id} value={c.id}>
                  {c.config.algorithm} · {c.id.slice(-8)}
                </option>
              ))}
            </select>
          </label>
          <button
            disabled={busy || !candidate}
            onClick={() =>
              run(async () =>
                setReady(
                  await api(
                    "/readiness?experiment_id=" + encodeURIComponent(candidate),
                  ),
                ),
              )
            }
          >
            Check readiness
          </button>
          {!candidate && (
            <small className="gm-reason">
              Choose a candidate to check its final evaluations and evidence.
            </small>
          )}
          {ready && (
            <>
              <p className={ready.ready ? "ws-notice" : "error"}>
                {ready.reason}
              </p>
              {!ready.ready && (
                <p>
                  {readinessAdvice(ready.reason)}{" "}
                  <a href={artifactLink("evaluation", candidate)}>
                    Open this candidate in Evaluation →
                  </a>{" "}
                  · <a href="/datasets">Prepare a corrected dataset →</a>
                </p>
              )}
              {ready.comparisons?.map((comparison: Item) => (
                <div key={comparison.split}>
                  <h3>{splitName[comparison.split]}</h3>
                  <Metrics value={comparison.candidate} />
                  <p>
                    Fixed baseline:{" "}
                    {Math.round(comparison.baseline.exact * 100)}% fully correct
                    cases.
                  </p>
                </div>
              ))}
              {ready.ready && (
                <>
                  <label className="ws-check">
                    <input
                      type="checkbox"
                      checked={approved}
                      onChange={(e) => setApproved(e.target.checked)}
                    />
                    I approve activating this candidate for reviewed recall.
                  </label>
                  <button
                    disabled={busy || !approved}
                    onClick={() =>
                      run(async () => {
                        await api("/promote", {
                          experiment_id: candidate,
                          evaluation_ids: ready.evaluation_ids,
                          actor: "paired-reviewer",
                        });
                        setApproved(false);
                        setReady(null);
                      }, "Reviewed recall deployment changed. The world learner is unchanged.")
                    }
                  >
                    Deploy approved candidate
                  </button>
                  {!approved && (
                    <small className="gm-reason">
                      Read and select the approval above to enable deployment.
                    </small>
                  )}
                </>
              )}
            </>
          )}
        </section>
      </div>
      <section className="ws-card">
        <h2>Try approved recall</h2>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            run(async () => setAnswer(await api("/recall", { query })));
          }}
        >
          <label>
            Question
            <input
              value={query}
              required
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
          <button disabled={busy || !health?.healthy}>
            Recall with evidence
          </button>
          {!health?.healthy && (
            <small className="gm-reason">
              Reviewed recall needs a healthy, explicitly approved deployment.
              Check readiness or resolve the health message above.
            </small>
          )}
        </form>
        {answer && (
          <>
            <p>
              {answer.insufficient_evidence
                ? "Insufficient evidence."
                : answer.uncertain
                  ? "Conflicting or uncertain evidence."
                  : "Retrieved evidence:"}
            </p>
            {answer.hits.map((h: Item, i: number) => (
              <div className="ws-inspect" key={i}>
                <p>{h.text}</p>
                <small>
                  {h.status} · {h.evidence_ids.map(short).join(", ")}
                </small>
              </div>
            ))}
          </>
        )}
      </section>
      <details className="gm-disclosure">
        <summary>Deployment history & rollback</summary>
        <p>
          Choose a prior deployment to inspect it. Rollback rechecks source
          availability, label revisions, and running code.
        </p>
        <ArtifactHistory
          kind="deployment"
          onSelect={(v) => {
            setTarget(v);
            setRollbackApproved(false);
          }}
        />
        {target && (
          <>
            <label className="ws-check">
              <input
                type="checkbox"
                checked={rollbackApproved}
                onChange={(e) => setRollbackApproved(e.target.checked)}
              />
              I approve restoring {short(target.id)} for reviewed recall.
            </label>
            <button
              disabled={busy || !rollbackApproved}
              onClick={() =>
                run(async () => {
                  await api("/rollback", {
                    deployment_id: target.id,
                    actor: "paired-reviewer",
                  });
                  setRollbackApproved(false);
                  setTarget(null);
                }, "Rollback saved as a new deployment event.")
              }
            >
              Restore approved deployment
            </button>
            {!rollbackApproved && (
              <small className="gm-reason">
                Read and select the rollback approval to restore this
                deployment.
              </small>
            )}
          </>
        )}
      </details>
    </>
  );
}
