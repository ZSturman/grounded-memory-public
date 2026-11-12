/** Presentation-only guidance. Visiting help never changes evidence or approvals. */
export const guides: Record<
  string,
  { input: string; output: string; steps: string[]; tip: string; done: string }
> = {
  overview: {
    input: "Your choice of goal",
    output: "A clear starting point for learning or reviewed recall",
    steps: [
      "Choose Observe and teach to start with a camera or recording, or Review and test memories to start with a note or saved evidence.",
      "Follow the steps on that screen. Every action explains what it saves and where to continue.",
      "Return here to switch goals or open any workspace directly.",
    ],
    tip: "All workspaces use your paired local Mac and one evidence history. Live learning does not require completing the review, experiment, or deployment stages.",
    done: "You can describe your chosen path and open its first screen.",
  },
  collection: {
    input: "Saved observations, teaching answers, or a note",
    output: "Unreviewed records in Labeling",
    steps: [
      "Add a note below, or collect inputs already saved by capture.",
      "Inspect the recent records to check what arrived.",
      "Open Labeling and review the original evidence.",
    ],
    tip: "Capture learns from observations and teaching as they arrive. Collection brings saved evidence into a separate review process. Automatic collection only runs while this page stays open.",
    done: "A record in the labeling queue is collected, but still needs a human review.",
  },
  labeling: {
    input: "Collected evidence and an editable suggestion",
    output: "Human-reviewed memories eligible for dataset checks",
    steps: [
      "Choose a record and inspect its original evidence.",
      "Correct the proposed memory, name the fact, and choose how to treat it.",
      "Save human review. Continue the queue, or open Datasets.",
    ],
    tip: "Suggestions are starting points, not confirmed facts. Skip keeps a record for later; Flag marks a problem. Neither confirms a label. Model-derived material never becomes independent evidence through review.",
    done: "Saving preserves the original and earlier reviews. Dataset eligibility is checked separately.",
  },
  datasets: {
    input: "Eligible human-reviewed evidence",
    output: "A fixed dataset version with reviewed questions",
    steps: [
      "Select reviewed records, then group the selected evidence.",
      "Assign independent groups to validation or held-out splits.",
      "Write independent questions and identify the evidence each should find.",
      "Name and prepare a version, then use that version in Experiments.",
    ],
    tip: "A dataset is a fixed collection of evidence and questions. Validation helps you choose settings. Held-out final and regression questions check your frozen choice; keep them out of tuning. One example can teach the workflow but cannot establish deployment readiness.",
    done: "Preparing pins the source and label versions. Corrections require a new version.",
  },
  experiments: {
    input: "A prepared dataset version",
    output: "A frozen search candidate",
    steps: [
      "Choose a prepared dataset, or follow the link from Datasets.",
      "Choose how to match questions to memories. Defaults are ready to try.",
      "Freeze the candidate, then validate that exact candidate.",
    ],
    tip: "A candidate is a saved search configuration tied to one dataset. Freezing does not train a model or activate it for recall. Compare settings using validation questions first.",
    done: "The candidate is saved for comparison; evaluation is still required.",
  },
  evaluation: {
    input: "A frozen candidate and its dataset's questions",
    output: "Comparison results for the selected evidence split",
    steps: [
      "Run validation and inspect the matches, misses, and failures.",
      "If needed, change settings in Experiments and validate a new candidate.",
      "Freeze your choice before running final evaluation and regression.",
      "Open Deployment for this candidate and check readiness.",
    ],
    tip: "Validation is for choosing settings. Final evaluation and regression use held-out questions. A saved run is not a passing result: readiness checks both final suites, category coverage, baseline comparison, and evidence compatibility.",
    done: "Inspect the result and the remaining readiness requirements before approving anything.",
  },
  deployment: {
    input: "An evaluated candidate",
    output: "Explicitly approved reviewed-memory recall",
    steps: [
      "Choose the candidate and check readiness. This check makes no changes.",
      "Resolve any missing or failing evaluations shown below.",
      "If ready, read the approval and explicitly deploy the candidate.",
      "Try a question against approved recall and inspect the evidence.",
    ],
    tip: "Reviewed recall searches the evidence you reviewed and approved. Deployment changes that search configuration; live world learning and perception keep their own behavior. Rollback also needs explicit approval.",
    done: "The active configuration and a healthy recall status confirm availability.",
  },
  capture: {
    input: "Camera, microphone, or a saved video",
    output: "Live learning, saved observations, and teaching history",
    steps: [
      "Choose Live camera or Import video. For video, upload a file and begin observation.",
      "Observe and answer questions in your own words. An unknown place is fine.",
      "Stop recording when finished and wait for saving to complete. Imported videos finish at the end of playback.",
      "Inspect beliefs below, or open Collection to import saved evidence for human review.",
    ],
    tip: "The live learner can learn while you observe. The separate Collection → Labeling workflow prepares reviewed recall. Later saves a question without stopping observation; you can answer saved questions afterward.",
    done: "Saved evidence remains available for inspection. Collection is a separate, explicit import step.",
  },
};

export function nextStep(status: {
  counts: Record<string, number>;
  awaiting_review: number;
  reviewed: number;
  deployed?: unknown;
}) {
  const c = status.counts;
  if (!c.raw)
    return {
      stage: "collection",
      reason: "Add a note or bring in saved capture evidence to begin.",
    };
  if (status.awaiting_review > 0)
    return {
      stage: "labeling",
      reason: `${status.awaiting_review} collected records still need a human decision. You can also continue existing work in any stage.`,
    };
  if (!c.dataset)
    return {
      stage: "datasets",
      reason:
        "Inspect eligible reviewed evidence and prepare a dataset with independent questions.",
    };
  if (!c.experiment)
    return {
      stage: "experiments",
      reason: "Choose a prepared dataset and freeze a search candidate.",
    };
  return {
    stage: status.deployed ? "deployment" : "evaluation",
    reason: status.deployed
      ? "Inspect the active configuration and its current health before trying reviewed recall."
      : "Select a candidate and inspect its evaluations. Saved runs alone do not establish readiness.",
  };
}

export const preferenceKey = "gm.guidance.v1";
export function readinessAdvice(reason: string): string {
  if (reason.includes("independent test AND"))
    return "Run Final evaluation and Regression for this candidate in Evaluation. If its dataset has no questions in those held-out splits, prepare a new dataset version with independent held-out groups and questions, then freeze and evaluate a new candidate.";
  if (reason.includes("seven evaluation categories"))
    return "Both held-out suites need questions covering accuracy, retrieval, temporal, contradiction, consolidation, grounding, and privacy. Add independently reviewed questions in a new dataset version, then freeze and evaluate a new candidate.";
  if (/full deployment corpus|curated final cases|baseline/.test(reason))
    return "Inspect the failed questions in Evaluation. Return incorrect evidence to Labeling, prepare corrected evidence as a new dataset version, and evaluate a new candidate. Keep held-out questions out of tuning.";
  if (/code|changed|revision|excluded|unavailable|stale/i.test(reason))
    return "Inspect the evidence and the reported change. Prepare a compatible dataset if needed, freeze a new candidate, and rerun its evaluations against the current code and evidence.";
  return "Inspect this candidate’s evaluation results and source evidence. Resolve the reported requirement, then check readiness again. Preparing corrected evidence requires a new dataset version.";
}
type PreferenceStore = {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
};
export function readHiddenTips(storage: PreferenceStore): string[] {
  try {
    const value: unknown = JSON.parse(storage.getItem(preferenceKey) || "[]");
    return Array.isArray(value)
      ? value.filter((v): v is string => typeof v === "string")
      : [];
  } catch {
    return [];
  }
}
export function writeHiddenTips(storage: PreferenceStore, hidden: string[]) {
  try {
    storage.setItem(preferenceKey, JSON.stringify(hidden));
  } catch {
    /* Help still works for this visit. */
  }
}

export function artifactLink(
  stage: "labeling" | "experiments" | "evaluation" | "deployment",
  id: string,
) {
  const key =
    stage === "labeling"
      ? "record"
      : stage === "experiments"
        ? "dataset"
        : "candidate";
  return `/${stage}?${key}=${encodeURIComponent(id)}`;
}

export async function resolveArtifactLink(
  id: string,
  kind: "dataset" | "experiment",
  fetchArtifact: (path: string) => Promise<any>,
) {
  if (
    !id.startsWith(kind + ":") ||
    id.length > 200 ||
    id.length <= kind.length + 1
  )
    throw new Error(
      "This link does not identify a valid " + kind + ". Choose one below.",
    );
  const item = await fetchArtifact("/artifact?id=" + encodeURIComponent(id));
  const valid =
    item?.id === id &&
    (kind === "dataset"
      ? item.spec && Array.isArray(item.records)
      : item.dataset_id && item.config);
  if (!valid)
    throw new Error(
      "This link points to a different kind of item. Choose one below.",
    );
  return kind === "dataset"
    ? { ...item, name: item.spec.name, records: item.records.length }
    : item;
}
