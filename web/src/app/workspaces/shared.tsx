"use client";
import { createContext, useContext, useEffect, useRef, useState } from "react";
import type { Notice } from "./Guidance";
import { resolveArtifactLink } from "./guideModel";
export type Item = Record<string, any>;
export const stages = [
  ["overview", "Overview", "Your next useful step"],
  ["collection", "Collection", "Bring experience into the pipeline"],
  ["labeling", "Labeling", "Turn evidence into reviewed memory"],
  ["datasets", "Datasets", "Prepare a trustworthy collection"],
  ["experiments", "Experiments", "Compare ways of finding memories"],
  ["evaluation", "Evaluation", "See what works and what is missed"],
  ["deployment", "Deployment", "Choose what powers reviewed recall"],
];
export const sourceName = (value: string) =>
  ({
    observations: "Earlier observations",
    world_observations: "World observations",
    world_teachings: "Teaching answers",
    utterances: "Earlier teaching",
    "user-note": "Notes",
  })[value] || value;
export const date = (stamp: number) =>
  stamp > 1000000000
    ? new Date(stamp * 1000).toLocaleString()
    : `Source clock: ${stamp}`;
export const short = (value: string) =>
  value?.length > 32 ? value.slice(0, 22) + "…" + value.slice(-6) : value;
export async function api(path: string, body?: unknown): Promise<any> {
  const r = await fetch(
    path.startsWith("/api/") ? path : "/api/v2/pipeline" + path,
    {
      method: body === undefined ? "GET" : "POST",
      headers:
        body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    },
  );
  if (!r.ok) {
    if (r.status === 401) window.dispatchEvent(new Event("gm-unpaired"));
    const text = await r.text();
    let detail = text;
    try {
      const parsed = JSON.parse(text);
      detail =
        typeof parsed.detail === "string"
          ? parsed.detail
          : JSON.stringify(parsed.detail);
    } catch {}
    throw new Error(
      detail || "Could not complete this action. Try refreshing.",
    );
  }
  return r.json();
}
type Context = {
  revision: number;
  busy: boolean;
  run: (fn: () => Promise<any>, message?: string | Notice) => Promise<boolean>;
  report: (message: string, kind?: "load" | "action") => void;
};
export const Work = createContext<Context>({
  revision: 0,
  busy: false,
  run: async () => false,
  report: () => {},
});
export const useWork = () => useContext(Work);
export function useLoad(path: string) {
  const { revision, report } = useWork();
  const [data, setData] = useState<any>(null);
  useEffect(() => {
    let current = true;
    setData(null);
    api(path)
      .then((v) => {
        if (current) setData(v);
      })
      .catch((e) => {
        if (current) report(e.message);
      });
    return () => {
      current = false;
    };
  }, [path, revision]);
  return data;
}
/** Resolve an explicit handoff separately from the newest-100 selector list. GETs only. */
export function useArtifactOptions(
  kind: "dataset" | "experiment",
  parameter: string,
) {
  const items = useLoad(`/items/${kind}?limit=100`);
  const [linked, setLinked] = useState<Item | null>(null);
  const [selected, setSelected] = useState("");
  const manualSelection = useRef(false);
  const [linkError, setLinkError] = useState("");
  useEffect(() => {
    let current = true;
    const id = new URLSearchParams(window.location.search).get(parameter);
    if (id)
      resolveArtifactLink(id, kind, api)
        .then((item) => {
          if (current) {
            setLinked(item);
            if (!manualSelection.current) setSelected(item.id);
          }
        })
        .catch(() => {
          if (current && !manualSelection.current)
            setLinkError(
              `The linked ${kind === "dataset" ? "dataset" : "candidate"} is unavailable or invalid. Choose another below, or return to ${kind === "dataset" ? "Datasets" : "Experiments"}.`,
            );
        });
    return () => {
      current = false;
    };
  }, [kind, parameter]);
  const options: Item[] =
    linked && !items?.some((item: Item) => item.id === linked.id)
      ? [linked, ...(items || [])]
      : items || [];
  return {
    options,
    selected,
    setSelected: (id: string) => {
      manualSelection.current = true;
      setSelected(id);
      setLinkError("");
    },
    linkError,
    loading: items === null,
  };
}
export function Empty({ children }: { children: React.ReactNode }) {
  return <p className="ws-empty">{children}</p>;
}
export function Details({
  value,
  title = "Technical details",
}: {
  value: unknown;
  title?: string;
}) {
  return (
    <details>
      <summary>{title}</summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}
export function Metrics({ value }: { value: Item }) {
  return (
    <div className="ws-metrics">
      {[
        ["precision", "Relevant matches"],
        ["recall", "Expected evidence found"],
        ["exact", "Fully correct cases"],
      ].map(([key, label]) => (
        <div key={key}>
          <strong>{Math.round((value?.[key] || 0) * 100)}%</strong>
          <span>{label}</span>
        </div>
      ))}
    </div>
  );
}
export function Pager({
  offset,
  total,
  onChange,
}: {
  offset: number;
  total: number;
  onChange: (n: number) => void;
}) {
  return (
    <div className="ws-toolbar">
      <button
        className="secondary"
        disabled={!offset}
        onClick={() => onChange(Math.max(0, offset - 30))}
      >
        Previous
      </button>
      <span>
        {total
          ? `${offset + 1}–${Math.min(offset + 30, total)} of ${total}`
          : "0 records"}
      </span>
      <button
        className="secondary"
        disabled={offset + 30 >= total}
        onClick={() => onChange(offset + 30)}
      >
        Next
      </button>
    </div>
  );
}
export function ArtifactHistory({
  kind,
  onSelect,
}: {
  kind: string;
  onSelect?: (v: Item) => void;
}) {
  const [offset, setOffset] = useState(0);
  const data = useLoad(`/items/${kind}?limit=30&offset=${offset}`);
  const { run } = useWork();
  const [detail, setDetail] = useState<Item | null>(null);
  return (
    <>
      <div className="ws-list">
        {data?.map((row: Item) => (
          <button
            className="ws-record"
            key={row.id}
            onClick={() =>
              run(async () => {
                const value = await api(
                  "/artifact?id=" + encodeURIComponent(row.id),
                );
                setDetail(value);
                onSelect?.(value);
              })
            }
          >
            <strong>
              {row.name ||
                row.action ||
                row.split ||
                row.config?.algorithm ||
                kind}
            </strong>
            <span>{short(row.id)}</span>
            {row.records !== undefined && (
              <small>
                {row.records} records · {row.cases} questions
              </small>
            )}
            {row.splits && (
              <small>
                {Object.entries(row.splits)
                  .map(([key, value]) => `${key}: ${value}`)
                  .join(" · ")}
              </small>
            )}
            {row.config && (
              <small>
                Maximum {row.config.top_k} answers · minimum score{" "}
                {row.config.min_score} · dataset {short(row.dataset_id)}
              </small>
            )}
            {row.candidate && (
              <small>
                {Math.round(row.candidate.precision * 100)}% relevant ·{" "}
                {Math.round(row.candidate.recall * 100)}% found ·{" "}
                {Math.round(row.candidate.exact * 100)}% correct · candidate{" "}
                {short(row.experiment_id)}
              </small>
            )}
          </button>
        ))}
      </div>
      {data?.length === 0 && <Empty>No {kind} history yet.</Empty>}
      <div className="ws-toolbar">
        <button
          className="secondary"
          disabled={!offset}
          onClick={() => setOffset(offset - 30)}
        >
          Previous
        </button>
        <button
          className="secondary"
          disabled={!data || data.length < 30}
          onClick={() => setOffset(offset + 30)}
        >
          More history
        </button>
      </div>
      {detail && <Details value={detail} title="Inspect selected artifact" />}
    </>
  );
}

export function EvidenceMedia({ raw }: { raw: Item }) {
  const payload = raw.payload;
  const references = [
    payload.perception?.frame,
    payload.evidence?.frame,
    payload.evidence?.asset,
    ...(payload.evidence_refs || []),
  ];
  const assets = [
    ...new Set<string>(
      references.filter(
        (v): v is string =>
          typeof v === "string" &&
          !v.includes("/") &&
          /\.(png|jpe?g|webp|wav|m4a|mp3)$/i.test(v),
      ),
    ),
  ];
  return (
    <>
      {assets
        .slice(0, 4)
        .map((name) =>
          /\.(wav|m4a|mp3)$/i.test(name) ? (
            <audio
              key={name}
              controls
              preload="none"
              src={"/api/assets/" + encodeURIComponent(name)}
              aria-label="Original audio evidence"
            />
          ) : (
            <img
              key={name}
              className="ws-evidence-image"
              src={"/api/assets/" + encodeURIComponent(name)}
              alt="Retained source evidence"
            />
          ),
        )}
      {!!assets.length && (
        <small>
          Media is read from original capture storage. A moved or unavailable
          original may not load; its reference remains in provenance.
        </small>
      )}
    </>
  );
}
