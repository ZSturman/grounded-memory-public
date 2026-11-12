"use client";
import { useState, useEffect, useRef } from "react";
import WorldPanel from "./WorldPanel";
import PipelinePanel from "./PipelinePanel";
import {
  Feedback,
  Guidance,
  HelpControls,
  Notice,
} from "./workspaces/Guidance";
type Data = Record<string, any>;
async function api(path: string, body?: unknown, method?: string) {
  const r = await fetch("/api" + path, {
    method: method ?? (body ? "POST" : "GET"),
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    if (r.status === 401) window.dispatchEvent(new Event("gm-unpaired"));
    const text = await r.text();
    try {
      const value = JSON.parse(text);
      throw new Error(typeof value.detail === "string" ? value.detail : text);
    } catch (e) {
      throw e instanceof SyntaxError ? new Error(text) : e;
    }
  }
  return r.json();
}
const asset = (name: string) => "/api/assets/" + encodeURIComponent(name);
export default function App() {
  const [paired, setPaired] = useState(false),
    [code, setCode] = useState(""),
    [sessions, setSessions] = useState<Data[]>([]),
    [current, setCurrent] = useState<Data | null>(null),
    [memory, setMemory] = useState<Data[]>([]),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(""),
    [place, setPlace] = useState(""),
    [date, setDate] = useState(""),
    [answer, setAnswer] = useState(""),
    [name, setName] = useState(""),
    [categories, setCategories] = useState(""),
    [aliases, setAliases] = useState(""),
    [models, setModels] = useState<Data[]>([]),
    [model, setModel] = useState("google/gemma-4-12b-qat"),
    [collision, setCollision] = useState<Data | null>(null),
    [audioRef, setAudioRef] = useState<string | null>(null),
    [spoken, setSpoken] = useState(false),
    [listening, setListening] = useState(false),
    [history, setHistory] = useState<Data[]>([]),
    [frame, setFrame] = useState<Data | null>(null),
    [aspect, setAspect] = useState(16 / 9);
  const video = useRef<HTMLVideoElement>(null),
    stream = useRef<MediaStream | null>(null),
    peer = useRef<RTCPeerConnection | null>(null),
    recorder = useRef<MediaRecorder | null>(null),
    lastSpoken = useRef("");
  const [notice, setNotice] = useState<Notice | null>(null);
  async function act(fn: () => Promise<any>) {
    setError("");
    try {
      return await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  }
  async function refresh() {
    const [s, m] = await Promise.all([api("/sessions"), api("/memory")]);
    setSessions(s);
    setMemory(m);
  }
  useEffect(() => {
    const lost = () => setPaired(false);
    window.addEventListener("gm-unpaired", lost);
    api("/status")
      .then((s) => setPaired(s.paired))
      .catch((e) => setError(String(e)));
    return () => window.removeEventListener("gm-unpaired", lost);
  }, []);
  useEffect(() => {
    if (paired) {
      refresh();
      api("/models")
        .then(setModels)
        .catch(() => {});
    }
  }, [paired]);
  useEffect(() => {
    if (!current) return;
    const timer = setInterval(
      () =>
        api("/sessions/" + current.id)
          .then(setCurrent)
          .catch((e) => setError(String(e))),
      1500,
    );
    return () => clearInterval(timer);
  }, [current?.id]);
  useEffect(() => {
    const q = current?.question;
    if (spoken && q && lastSpoken.current !== q.id) {
      const voices = speechSynthesis.getVoices().filter((v) => v.localService);
      if (voices.length) {
        const u = new SpeechSynthesisUtterance(q.text);
        u.voice = voices.find((v) => v.lang.startsWith("en")) ?? voices[0];
        api("/v2/output", {
          duration_seconds: Math.min(240, q.text.length / 10 + 3),
        })
          .then(() => {
            speechSynthesis.speak(u);
            lastSpoken.current = q.id;
          })
          .catch((e) => setError(String(e)));
      } else {
        fetch("/api/speech", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: q.text }),
        })
          .then((r) => {
            if (!r.ok) throw new Error("Local speech is unavailable");
            return r.blob();
          })
          .then(async (b) => {
            await api("/v2/output", {
              duration_seconds: Math.min(240, q.text.length / 10 + 3),
            });
            const url = URL.createObjectURL(b);
            const a = new Audio(url);
            a.onended = () => URL.revokeObjectURL(url);
            a.play().catch(() =>
              setError(
                "Tap Speak questions again to allow local audio playback.",
              ),
            );
            lastSpoken.current = q.id;
          })
          .catch((e) => setError(String(e)));
      }
    }
  }, [current?.question?.id, spoken]);
  async function create(mode: string) {
    const episode = await api("/v2/episodes", { mode });
    const s = await api("/sessions/" + episode.id);
    setCurrent(s);
    setFrame(null);
    setHistory([]);
    setCollision(null);
    setPlace("");
    setNotice({
      text:
        mode === "live"
          ? "Live experience created. Connect your camera below to begin observing."
          : "Video experience created. Choose a recording below, then begin observation.",
    });
    await refresh();
    return s;
  }
  async function context(resolution?: string, previous_id?: string) {
    if (!current) return;
    const r = await api("/sessions/" + current.id + "/context", {
      label: place || "Unknown place",
      unknown: !place,
      capture_time: date || null,
      relative_time: !date,
      resolution,
      previous_id,
    });
    if (r.needs_clarification) setCollision({ ...r, kind: "place" });
    else {
      setCurrent(r);
      setCollision(null);
      refresh();
    }
  }
  async function upload(file: File) {
    setBusy("Importing recording…");
    let s = current;
    if (!s || s.mode !== "recall") s = await create("recall");
    for (let offset = 0; offset < file.size;) {
      const chunk = file.slice(offset, offset + 4 * 1024 * 1024);
      const r = await fetch(
        "/api/sessions/" + s!.id + "/upload?offset=" + offset,
        { method: "PUT", body: chunk },
      );
      if (!r.ok) throw new Error(await r.text());
      offset = (await r.json()).next_offset;
      setBusy("Importing " + Math.round((offset / file.size) * 100) + "%");
    }
    setBusy("Preparing playback…");
    setCurrent(await api("/sessions/" + s!.id + "/upload/complete", {}));
    setNotice({
      text: "Recording uploaded. Choose Begin observation below to let the learner inspect it.",
    });
    await refresh();
  }
  async function camera() {
    setBusy("Connecting camera…");
    if (!window.isSecureContext)
      throw new Error(
        "Phone cameras require HTTPS. Run make phone and follow certificate setup.",
      );
    let s = current;
    if (!s || s.mode !== "live") s = await create("live");
    stream.current = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 15 },
        facingMode: { ideal: "environment" },
      },
      audio: true,
    });
    if (video.current) {
      video.current.srcObject = stream.current;
      await video.current.play();
    }
    const pc = new RTCPeerConnection({ iceServers: [] });
    peer.current = pc;
    stream.current.getTracks().forEach((t) => pc.addTrack(t, stream.current!));
    await pc.setLocalDescription(await pc.createOffer());
    await new Promise<void>((resolve, reject) => {
      if (pc.iceGatheringState === "complete") return resolve();
      const timeout = setTimeout(
        () => reject(new Error("Camera negotiation timed out")),
        10000,
      );
      pc.onicegatheringstatechange = () => {
        if (pc.iceGatheringState === "complete") {
          clearTimeout(timeout);
          resolve();
        }
      };
    });
    const response = await api(
      "/sessions/" + s!.id + "/offer",
      pc.localDescription,
    );
    await pc.setRemoteDescription(response);
    setCurrent(await api("/sessions/" + s!.id));
    setNotice({
      text: "Camera connected. Observe and answer questions below; the place can remain unknown.",
    });
    await refresh();
  }
  async function control(action: string, region_id?: string) {
    if (!current) return;
    if (action === "stop") {
      peer.current?.close();
      stream.current?.getTracks().forEach((t) => t.stop());
    }
    setCurrent(
      await api("/sessions/" + current.id + "/control", { action, region_id }),
    );
    if (action === "stop")
      setNotice({
        text: "Observation stopped. Inspect this experience and its learning below, or import its saved evidence into review.",
        action: { href: "/collection", label: "Open Collection" },
      });
  }
  async function interpret() {
    setBusy("Reading your answer locally…");
    const result = await api("/interpret", { text: answer, model });
    setName(result.label);
    setCategories(result.categories.join(", "));
    setAliases(result.aliases.join(", "));
  }
  async function teach(resolution?: string, previous_id?: string) {
    if (!current?.question) return;
    const result = await api("/sessions/" + current.id + "/teach", {
      question_id: current.question.id,
      text: answer,
      label: name,
      categories: categories
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      aliases: aliases
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      resolution,
      previous_id,
      audio_ref: audioRef,
    });
    if (result.needs_clarification) setCollision({ ...result, kind: "object" });
    else {
      setCollision(null);
      setAnswer("");
      setName("");
      setCategories("");
      setAliases("");
      setCurrent(await api("/sessions/" + current.id));
      await refresh();
    }
  }
  async function voice() {
    if (listening) {
      recorder.current?.stop();
      setListening(false);
      return;
    }
    speechSynthesis.cancel();
    const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
    const chunks: BlobPart[] = [];
    const r = new MediaRecorder(mic);
    recorder.current = r;
    r.ondataavailable = (e) => chunks.push(e.data);
    r.onstop = () => {
      mic.getTracks().forEach((t) => t.stop());
      act(async () => {
        setBusy("Transcribing locally…");
        const form = new FormData();
        form.append("file", new Blob(chunks, { type: r.mimeType }), "reply");
        const response = await fetch("/api/transcribe", {
          method: "POST",
          body: form,
        });
        if (!response.ok) throw new Error(await response.text());
        const value = await response.json();
        setAnswer(value.text);
        setAudioRef(value.audio_ref);
      });
    };
    r.start();
    setListening(true);
  }
  async function selectSession(s: Data) {
    setCurrent(s);
    setFrame(null);
    setCollision(null);
    setHistory(await api("/sessions/" + s.id + "/observations"));
  }
  const latest = frame?.perception ?? current?.latest,
    q = current?.question;
  if (!paired)
    return (
      <main>
        <div className="card pair">
          <div className="eyebrow">Local connection</div>
          <h1>Grounded Memory</h1>
          <a href="/">Open pipeline applications →</a>
          <p>Observe the world. Ask a person. Remember the evidence.</p>
          <ol>
            <li>
              Run <code>make dev</code> on your Mac and keep it running.
            </li>
            <li>Enter the printed pairing code.</li>
            <li>Choose Live camera or Import video, then follow the steps.</li>
          </ol>
          <label>
            Pairing code shown in your Mac terminal
            <input
              value={code}
              onChange={(e) => setCode(e.target.value)}
              inputMode="numeric"
            />
          </label>
          <button
            onClick={() =>
              act(async () => {
                await api("/pair", { code });
                setPaired(true);
              })
            }
          >
            Connect this browser
          </button>
          {error && <p role="alert">{error}</p>}
        </div>
      </main>
    );
  return (
    <main>
      <header>
        <div>
          <div className="eyebrow">Observe · predict · ask · revise</div>
          <h1>Grounded Memory</h1>
          <a href="/">Open pipeline applications →</a>
          <span className="muted">Your experience, learned locally.</span>
        </div>
        <div className="row">
          <button
            className="secondary"
            onClick={() => act(() => create("recall"))}
          >
            Import video
          </button>
          <button onClick={() => act(() => create("live"))}>Live camera</button>
        </div>
      </header>
      <HelpControls />
      <Guidance stage="capture" />
      {notice && <Feedback notice={notice} dismiss={() => setNotice(null)} />}
      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}
      {busy && (
        <div className="progress" role="status">
          {busy}
        </div>
      )}
      <div
        className={
          "grid" +
          (!current || current.world_version === 2
            ? " world-layout"
            : q
              ? " has-question"
              : "")
        }
      >
        <aside className="sidebar">
          <h2>Experiences</h2>
          <p className="muted">Saved recordings and learning sessions</p>
          {sessions.map((s) => (
            <button
              key={s.id}
              className={"session " + (s.id === current?.id ? "active" : "")}
              onClick={() => act(() => selectSession(s))}
            >
              <strong>{s.context?.label ?? "New experience"}</strong>
              <br />
              <small>
                {s.mode} · {s.status}
                <br />
                {new Date(s.created_at * 1000).toLocaleString()}
              </small>
            </button>
          ))}
        </aside>
        <section>
          <PipelinePanel enabled={paired} />
          <WorldPanel
            episodeId={current?.id}
            enabled={paired && (!current || current.world_version === 2)}
          />
          <div className="card">
            <div className="row">
              <h2>
                {current?.context?.label ?? "Observing an unfamiliar place"}
              </h2>
              {current && (
                <span className="badge">
                  {current.mode} · {current.status}
                </span>
              )}
            </div>
            <div className="stage">
              {/* Sized to the media's aspect ratio so region percentages align with pixels. */}
              <div
                className="media"
                style={{
                  aspectRatio: aspect,
                  width: `min(100%, ${Math.round(480 * aspect)}px)`,
                }}
              >
                {current?.mode === "live" ? (
                  <video
                    ref={video}
                    autoPlay
                    playsInline
                    muted
                    onLoadedMetadata={(e) =>
                      setAspect(
                        e.currentTarget.videoWidth /
                          e.currentTarget.videoHeight || 16 / 9,
                      )
                    }
                  />
                ) : latest?.frame ? (
                  <img
                    src={asset(latest.frame)}
                    alt="Latest observed frame"
                    onLoad={(e) =>
                      setAspect(
                        e.currentTarget.naturalWidth /
                          e.currentTarget.naturalHeight,
                      )
                    }
                  />
                ) : current?.preview ? (
                  <video
                    src={asset(current.preview)}
                    controls
                    playsInline
                    onLoadedMetadata={(e) =>
                      setAspect(
                        e.currentTarget.videoWidth /
                          e.currentTarget.videoHeight || 16 / 9,
                      )
                    }
                  />
                ) : (
                  <p>Connect a camera or choose a recording.</p>
                )}
                {latest?.regions?.map((r: Data) => (
                  <button
                    aria-label={
                      r.matches?.[0]?.label ?? "Ask about this region"
                    }
                    title={r.matches?.[0]?.label ?? "Unknown region"}
                    key={r.id}
                    className="outline"
                    style={{
                      left: r.bbox[0] * 100 + "%",
                      top: r.bbox[1] * 100 + "%",
                      width: r.bbox[2] * 100 + "%",
                      height: r.bbox[3] * 100 + "%",
                    }}
                    onClick={() => act(() => control("select", r.id))}
                  >
                    {r.matches?.[0] && (
                      <span className="tag">
                        {r.matches[0].label}{" "}
                        {Math.round(r.matches[0].appearance * 100)}%
                      </span>
                    )}
                  </button>
                ))}
              </div>
            </div>
            {latest && (
              <p className="muted">
                <small>
                  {latest.media_time?.toFixed(2)}s · {latest.encoder} ·{" "}
                  {latest.segmentation}
                </small>
              </p>
            )}
            {current?.mode === "live" ? (
              <div className="row">
                <button onClick={() => act(camera)}>
                  Connect phone / webcam
                </button>
                <button
                  className="secondary"
                  onClick={() => act(() => control("stop"))}
                >
                  Stop recording
                </button>
              </div>
            ) : (
              <div className="file">
                <label>
                  Choose MP4, MOV, or WebM
                  <input
                    type="file"
                    accept="video/*"
                    disabled={!!busy}
                    onChange={(e) =>
                      e.target.files?.[0] &&
                      act(() => upload(e.target.files![0]))
                    }
                  />
                </label>
                {current?.source && (
                  <button
                    disabled={!!busy}
                    onClick={() =>
                      act(async () =>
                        setCurrent(
                          await api("/sessions/" + current.id + "/start", {}),
                        ),
                      )
                    }
                  >
                    Begin observation
                  </button>
                )}
              </div>
            )}
            {current?.preview &&
              (current.mode === "recall" || current.status === "stopped") && (
                <video
                  style={{ width: "100%" }}
                  src={asset(current.preview)}
                  controls
                />
              )}
          </div>
          {current && !current.context && current.world_version !== 2 && (
            <div className="card">
              <h2>First, where and when?</h2>
              <label>
                What place is this?
                <input
                  value={place}
                  onChange={(e) => setPlace(e.target.value)}
                  placeholder="Kitchen at home — or leave unknown"
                />
              </label>
              <label>
                Capture date and time, if known
                <input
                  type="datetime-local"
                  value={date}
                  onChange={(e) => setDate(e.target.value)}
                />
              </label>
              <p className="muted">
                Without a date, we use the recording’s relative timeline.
              </p>
              <button onClick={() => act(() => context())}>
                Establish context
              </button>
            </div>
          )}
          <div className="card">
            <h2>Recall timeline</h2>
            <button
              className="secondary"
              onClick={() =>
                current &&
                act(async () =>
                  setHistory(
                    await api("/sessions/" + current.id + "/observations"),
                  ),
                )
              }
            >
              Load saved observations
            </button>
            {history.length > 0 && (
              <>
                <input
                  type="range"
                  min={0}
                  max={history.length - 1}
                  defaultValue={0}
                  onChange={(e) => setFrame(history[Number(e.target.value)])}
                />
                <p>
                  {history.length} saved observations. Frames are kept every few
                  seconds and for every question; the recording holds the rest.
                  Browsing does not add learning evidence.
                </p>
              </>
            )}
          </div>
          <div className="card">
            <h2>What has been learned</h2>
            {memory.length === 0 && (
              <p className="muted">
                Names and categories appear here after your teaching.
              </p>
            )}
            {memory.map((e) => (
              <div className="memory" key={e.id}>
                {e.examples?.at(-1)?.asset && (
                  <img
                    src={asset(e.examples.at(-1).asset)}
                    alt="Remembered evidence"
                  />
                )}
                <strong>{e.label}</strong>{" "}
                <span className="badge">{e.kind}</span>
                <p>
                  <small>{e.examples.length} appearance examples</small>
                </p>
                {e.claims.map((c: Data) => (
                  <div key={c.id}>
                    <small>
                      {c.predicate}:{" "}
                      {memory.find((m) => m.id === c.object)?.label ?? c.object}
                    </small>{" "}
                    <button
                      className="secondary"
                      style={{ padding: "3px 8px" }}
                      onClick={() =>
                        act(async () => {
                          await api("/claims/" + c.id + "/revoke", {});
                          refresh();
                        })
                      }
                    >
                      Undo claim
                    </button>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </section>
        <aside
          className="teaching"
          style={{
            display:
              !current || current.world_version === 2 ? "none" : undefined,
          }}
        >
          <div className="card">
            <h2>Learning together</h2>
            <label>
              Conversation model
              <select value={model} onChange={(e) => setModel(e.target.value)}>
                {!models.length && <option>{model}</option>}
                {models.map((m) => (
                  <option key={m.id}>{m.id}</option>
                ))}
              </select>
            </label>
            <label>
              <input
                type="checkbox"
                style={{ width: "auto" }}
                checked={spoken}
                onChange={(e) => setSpoken(e.target.checked)}
              />{" "}
              Speak questions with a local voice
            </label>
            {q ? (
              <>
                <img
                  className="crop"
                  src={asset(q.region.asset)}
                  alt="Object the system is asking about"
                />
                <h2>{q.text}</h2>
                <small>
                  Frozen evidence at {q.media_time.toFixed(2)}s ·{" "}
                  {q.context.label}
                </small>
                <label>
                  Your answer
                  <textarea
                    value={answer}
                    onChange={(e) => setAnswer(e.target.value)}
                    placeholder="That is a digital clock. It is also a clock."
                  />
                </label>
                <div className="row">
                  <button className="secondary" onClick={() => act(voice)}>
                    {listening ? "Finish reply" : "Speak reply"}
                  </button>
                  <button
                    disabled={!answer || !!busy}
                    onClick={() => act(interpret)}
                  >
                    Interpret answer
                  </button>
                </div>
                <p className="muted">
                  <small>
                    Review the transcript and proposed teaching before saving.
                  </small>
                </p>
                <label>
                  Name
                  <input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                  />
                </label>
                <label>
                  Categories, separated by commas
                  <input
                    value={categories}
                    onChange={(e) => setCategories(e.target.value)}
                  />
                </label>
                <label>
                  Other names
                  <input
                    value={aliases}
                    onChange={(e) => setAliases(e.target.value)}
                  />
                </label>
                <button
                  disabled={!name || !answer}
                  onClick={() => act(() => teach())}
                >
                  Confirm this teaching
                </button>
                <div className="row" style={{ marginTop: 10 }}>
                  <button
                    className="secondary"
                    onClick={() => act(() => control("skip"))}
                  >
                    Skip
                  </button>
                  <button
                    className="secondary"
                    onClick={() => act(() => control("later"))}
                  >
                    Ask later
                  </button>
                </div>
              </>
            ) : (
              <p className="muted">
                Use the world model panel to answer questions about places,
                sounds, events, and explanations. Tap a visual region to inspect
                its outline.
              </p>
            )}
            {current && (
              <button
                className="secondary"
                style={{ marginTop: 15 }}
                onClick={() =>
                  act(() => control(current.paused ? "resume" : "pause"))
                }
              >
                {current.paused ? "Resume questions" : "Pause questions"}
              </button>
            )}
          </div>
          {collision && (
            <div className="card">
              <h2>Same name. Same thing?</h2>
              <p>
                Compare the previous evidence. Are these the same physical{" "}
                {collision.kind}, or do they share a category?
              </p>
              {collision.matches.map((m: Data) => (
                <div key={m.id}>
                  <strong>{m.label}</strong>
                  {m.examples.at(-1)?.asset && (
                    <img
                      className="crop"
                      src={asset(m.examples.at(-1).asset)}
                      alt="Previous entity"
                    />
                  )}
                  {[
                    ["same", "Same physical entity"],
                    ["same_kind", "Different entity, same kind"],
                    ["subtype", "Different subtype, shared category"],
                    ["unrelated", "Unrelated, same name"],
                  ].map(([choice, label]) => (
                    <button
                      key={choice}
                      className="secondary"
                      style={{ margin: 4 }}
                      onClick={() =>
                        act(() =>
                          collision.kind === "place"
                            ? context(choice, m.id)
                            : teach(choice, m.id),
                        )
                      }
                    >
                      {label}
                    </button>
                  ))}
                </div>
              ))}
            </div>
          )}
        </aside>
      </div>
    </main>
  );
}
