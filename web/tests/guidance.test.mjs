import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// Use the installed compiler so this test also runs on the supported Node 20.
const source = await readFile(
  new URL("../src/app/workspaces/guideModel.ts", import.meta.url),
  "utf8",
);
const { outputText } = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ES2022,
    target: ts.ScriptTarget.ES2022,
  },
});
const {
  nextStep,
  readHiddenTips,
  writeHiddenTips,
  preferenceKey,
  artifactLink,
  resolveArtifactLink,
  readinessAdvice,
} = await import(
  "data:text/javascript;base64," + Buffer.from(outputText).toString("base64")
);

test("next steps follow actual availability without treating saved evaluations as passes", () => {
  const status = { counts: {}, reviewed: 0, awaiting_review: 0 };
  assert.equal(nextStep(status).stage, "collection");
  status.counts.raw = 3;
  status.awaiting_review = 2;
  assert.equal(nextStep(status).stage, "labeling");
  status.awaiting_review = 0;
  status.reviewed = 3;
  assert.equal(nextStep(status).stage, "datasets");
  status.counts.dataset = 1;
  assert.equal(nextStep(status).stage, "experiments");
  status.counts.experiment = 1;
  status.counts.evaluation = 99;
  assert.equal(nextStep(status).stage, "evaluation");
  assert.match(nextStep(status).reason, /do not establish readiness/);
  status.deployed = { id: "deployment:old" };
  assert.match(nextStep(status).reason, /current health/);
});

test("tip dismissal survives a reload and restoration only changes guidance preferences", () => {
  const values = new Map([["pairing", "untouched"]]);
  const storage = {
    getItem: (k) => values.get(k) ?? null,
    setItem: (k, v) => values.set(k, v),
  };
  assert.deepEqual(readHiddenTips(storage), []);
  writeHiddenTips(storage, ["tip.collection", "steps.collection"]);
  assert.deepEqual(readHiddenTips(storage), [
    "tip.collection",
    "steps.collection",
  ]);
  writeHiddenTips(storage, []);
  assert.deepEqual(readHiddenTips(storage), []);
  assert.equal(values.get("pairing"), "untouched");
  assert.deepEqual([...values.keys()], ["pairing", preferenceKey]);
});

test("unavailable or malformed preference storage cannot block help", () => {
  const blocked = {
    getItem() {
      throw new Error("blocked");
    },
    setItem() {
      throw new Error("quota");
    },
  };
  assert.deepEqual(readHiddenTips(blocked), []);
  assert.doesNotThrow(() => writeHiddenTips(blocked, ["tip.capture"]));
  for (const text of ["{broken", "null", "{}", "42"])
    assert.deepEqual(readHiddenTips({ ...blocked, getItem: () => text }), []);
  assert.deepEqual(
    readHiddenTips({ ...blocked, getItem: () => '["tip.capture", 1, null]' }),
    ["tip.capture"],
  );
});

test("handoffs encode the exact artifact without carrying actions or approval", () => {
  for (const [stage, key] of [
    ["labeling", "record"],
    ["experiments", "dataset"],
    ["evaluation", "candidate"],
    ["deployment", "candidate"],
  ]) {
    const url = new URL(
      artifactLink(stage, "note:x & ?#/+"),
      "http://localhost",
    );
    assert.equal(url.pathname, "/" + stage);
    assert.deepEqual([...url.searchParams.entries()], [[key, "note:x & ?#/+"]]);
  }
});

test("an older linked dataset resolves directly without relying on selector pagination", async () => {
  const calls = [];
  const item = await resolveArtifactLink(
    "dataset:older",
    "dataset",
    async (path) => {
      calls.push(path);
      return {
        id: "dataset:older",
        spec: { name: "Older reviewed version" },
        records: [{ id: "note:one" }],
      };
    },
  );
  assert.equal(item.name, "Older reviewed version");
  assert.equal(item.records, 1);
  assert.deepEqual(calls, ["/artifact?id=dataset%3Aolder"]);
});

test("a candidate link reads only the matching candidate", async () => {
  const candidate = {
    id: "experiment:old",
    dataset_id: "dataset:older",
    config: { algorithm: "coverage" },
  };
  assert.equal(
    await resolveArtifactLink(
      candidate.id,
      "experiment",
      async () => candidate,
    ),
    candidate,
  );
});

test("invalid, missing, and wrong-kind links cannot silently select an item", async () => {
  let calls = 0;
  const fetch = async () => {
    calls++;
    return { id: "dataset:wrong" };
  };
  for (const id of [
    "",
    "experiment:wrong",
    "dataset:",
    "dataset:" + "x".repeat(201),
  ])
    await assert.rejects(resolveArtifactLink(id, "dataset", fetch));
  assert.equal(calls, 0);
  await assert.rejects(
    resolveArtifactLink("dataset:missing", "dataset", async () => {
      throw new Error("not found");
    }),
  );
  await assert.rejects(resolveArtifactLink("dataset:wrong", "dataset", fetch));
  await assert.rejects(
    resolveArtifactLink("dataset:expected", "dataset", async () => ({
      id: "dataset:different",
      spec: {},
      records: [],
    })),
  );
});

test("readiness recovery explains missing suites, coverage, stale evidence, and failures", () => {
  assert.match(
    readinessAdvice(
      "Promotion needs independent test AND fixed regression evaluations",
    ),
    /no questions in those held-out splits/,
  );
  assert.match(
    readinessAdvice(
      "Both final suites must cover all seven evaluation categories",
    ),
    /accuracy, retrieval, temporal, contradiction, consolidation, grounding, and privacy/,
  );
  assert.match(
    readinessAdvice(
      "Candidate must pass all curated final cases and not regress against baseline",
    ),
    /failed questions/,
  );
  assert.match(
    readinessAdvice("Candidate code differs from the running code; reevaluate"),
    /new candidate/,
  );
  assert.match(
    readinessAdvice("An unexpected requirement"),
    /reported requirement/,
  );
});
