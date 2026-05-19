import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, Brain, Download, Filter, Info, Play, RefreshCw, Zap } from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

type Category =
  | "lexical"
  | "syntactic"
  | "persona"
  | "politeness"
  | "specificity"
  | "negation"
  | "position";

type Variant = {
  id: string;
  prompt: string;
  category: Category | "base";
  transform: string;
  system_prompt?: string | null;
};

type ResponseResult = {
  variant_id: string;
  output: string;
  latency_ms: number;
  error?: string | null;
};

type Point = {
  variant_id: string;
  x: number;
  y: number;
  cluster: number;
  semantic_distance: number;
  output_length: number;
  token_count: number;
  entropy: number;
};

type RunRecord = {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  request: {
    base_prompt: string;
    categories: Category[];
    sample_count: number;
    model?: string | null;
    temperature: number;
    seed: number;
    mode: "demo" | "mock" | "live" | "local";
  };
  variants: Variant[];
  responses: ResponseResult[];
  points: Point[];
  metrics?: {
    embedding_device: string;
    embedding_ms: number;
    projection_method: string;
    clustering_method: string;
    api_success_count: number;
    api_error_count: number;
    kl_divergence_available: boolean;
    category_summaries: { category: string; count: number; avg_distance: number; max_distance: number }[];
  } | null;
  error?: string | null;
};

type TrainingRecord = {
  status: "idle" | "running" | "completed" | "failed";
  started_at?: string | null;
  completed_at?: string | null;
  device: string;
  cuda_available: boolean;
  cuda_device_name: string;
  dataset_size: number;
  train_size: number;
  validation_size: number;
  epochs: number;
  final_loss?: number | null;
  validation_accuracy?: number | null;
  threshold?: number | null;
  training_ms?: number | null;
  model_path?: string | null;
  error?: string | null;
};

const categories: Category[] = ["lexical", "syntactic", "persona", "politeness", "specificity", "negation", "position"];
const colors = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#d97706", "#0891b2", "#be123c", "#4b5563"];
const categoryLabels: Record<Category, string> = {
  lexical: "Word changes",
  syntactic: "Sentence structure",
  persona: "Role framing",
  politeness: "Tone changes",
  specificity: "More specific",
  negation: "Do-not rules",
  position: "Instruction placement",
};

const categoryHelp: Record<Category, string> = {
  lexical: "Changes individual words to see if wording alone changes the answer.",
  syntactic: "Rephrases the sentence while keeping the same basic request.",
  persona: "Adds a role like expert, teacher, or reviewer before the prompt.",
  politeness: "Tests whether polite or direct wording changes the answer.",
  specificity: "Adds clearer constraints such as length, format, or required points.",
  negation: "Adds instructions with 'do not' to see if the model reacts differently.",
  position: "Moves instructions into different parts of the prompt setup.",
};

function App() {
  const [prompt, setPrompt] = useState("Explain how to safely roll out a new AI chatbot feature to production users.");
  const [mode, setMode] = useState<"demo" | "mock" | "live" | "local">("demo");
  const [sampleCount, setSampleCount] = useState(64);
  const [selectedCategories, setSelectedCategories] = useState<Category[]>(categories);
  const [run, setRun] = useState<RunRecord | null>(null);
  const [selectedId, setSelectedId] = useState("base");
  const [categoryFilter, setCategoryFilter] = useState<Category | "all">("all");
  const [busy, setBusy] = useState(false);
  const [trainingBusy, setTrainingBusy] = useState(false);
  const [training, setTraining] = useState<TrainingRecord | null>(null);
  const [health, setHealth] = useState<string>("checking");

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((res) => res.json())
      .then((data) => setHealth(data.live_ready ? `Gemini ready | ${data.dependencies.cuda_device}` : `demo ready | ${data.dependencies.cuda_device}`))
      .catch(() => setHealth("backend offline"));
    fetch(`${API_BASE}/api/training/risk-model`)
      .then((res) => res.json())
      .then((data) => setTraining(data))
      .catch(() => setTraining(null));
  }, []);

  useEffect(() => {
    if (!run || run.status === "completed" || run.status === "failed" || mode === "demo") return;
    const timer = window.setInterval(async () => {
      const next = await fetch(`${API_BASE}/api/runs/${run.id}`).then((res) => res.json());
      setRun(next);
      if (next.status === "completed") setSelectedId("base");
    }, 1500);
    return () => window.clearInterval(timer);
  }, [run, mode]);

  const variantById = useMemo(() => new Map(run?.variants.map((variant) => [variant.id, variant]) ?? []), [run]);
  const responseById = useMemo(() => new Map(run?.responses.map((response) => [response.variant_id, response]) ?? []), [run]);
  const pointById = useMemo(() => new Map(run?.points.map((point) => [point.variant_id, point]) ?? []), [run]);
  const selectedVariant = variantById.get(selectedId) ?? run?.variants[0];
  const selectedResponse = selectedVariant ? responseById.get(selectedVariant.id) : undefined;
  const baseResponse = responseById.get("base");
  const runError = run?.status === "failed" ? run.error ?? "Run failed." : null;
  const outputText = selectedResponse?.error || selectedResponse?.output || runError || "No output yet.";

  const filteredPoints = useMemo(() => {
    if (!run) return [];
    return run.points.filter((point) => {
      const variant = variantById.get(point.variant_id);
      return categoryFilter === "all" || variant?.category === categoryFilter || variant?.category === "base";
    });
  }, [run, categoryFilter, variantById]);

  async function startRun() {
    setBusy(true);
    const response = await fetch(`${API_BASE}/api/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        base_prompt: prompt,
        categories: selectedCategories,
        sample_count: sampleCount,
        temperature: 0.4,
        seed: 42,
        mode,
      }),
    });
    const next = await response.json();
    setRun(next);
    setSelectedId("base");
    setBusy(false);
  }

  async function trainRiskModel() {
    setTrainingBusy(true);
    setTraining((current) => current ? { ...current, status: "running" } : {
      status: "running",
      device: "unknown",
      cuda_available: false,
      cuda_device_name: "unknown",
      dataset_size: 0,
      train_size: 0,
      validation_size: 0,
      epochs: 0,
    });
    try {
      const response = await fetch(`${API_BASE}/api/training/risk-model`, { method: "POST" });
      const next = await response.json();
      setTraining(next);
    } finally {
      setTrainingBusy(false);
    }
  }

  function toggleCategory(category: Category) {
    setSelectedCategories((current) =>
      current.includes(category) ? current.filter((item) => item !== category) : [...current, category],
    );
  }

  const topShifts = [...(run?.points ?? [])]
    .filter((point) => point.variant_id !== "base")
    .sort((a, b) => b.semantic_distance - a.semantic_distance)
    .slice(0, 5);

  return (
    <main>
      <div className="sky-layer" />
      <div className="moving-clouds" aria-hidden="true">
        <span className="cloud cloud-one" />
        <span className="cloud cloud-two" />
        <span className="cloud cloud-three" />
      </div>
      <header className="topbar">
        <a className="brand" href="#">
          <span className="brand-mark">P</span>
          PromptMap
        </a>
        <div className="topbar-note">Prompt stability tester</div>
      </header>

      <section className="hero">
        <div className="eyebrow">{health}</div>
        <h1>
          Test How Stable Your <TooltipTerm text="AI Prompt" explanation="The instruction or question you give to an AI model." /> Is
        </h1>
        <p>
          Write one prompt, let the app create small variations, and see whether the AI keeps giving similar answers or changes too much.
        </p>
        <button className="hero-action" disabled={busy || selectedCategories.length === 0} onClick={startRun} type="button">
          {busy ? <RefreshCw size={18} /> : <Play size={18} />}
          Create a Stability Map
        </button>
      </section>

      <section className="workspace" id="mapper">
        <div className="dashboard-topline">
          <div>
            <span className="app-dot" />
            GPU Prompt Sensitivity Lab
          </div>
          <div className="status">{run ? `${run.status} | ${run.points.length || run.request.sample_count} variants` : "ready"}</div>
        </div>
        <aside className="controls">
          <label>
            <TooltipTerm text="Original prompt" explanation="The main prompt you want to test." />
            <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} />
          </label>
          <div className="row">
            <label>
              Mode
                <select value={mode} onChange={(event) => setMode(event.target.value as "demo" | "mock" | "live" | "local")}>
                  <option value="demo">Demo</option>
                  <option value="mock">Mock async</option>
                  <option value="live">Gemini live</option>
                  <option value="local">Local model (GPU)</option>
              </select>
            </label>
            <label>
              <TooltipTerm text="Number of tests" explanation="How many slightly changed versions of your prompt the app should try." />
              <input type="number" min={5} max={250} value={sampleCount} onChange={(event) => setSampleCount(Number(event.target.value))} />
            </label>
          </div>
          <div className="category-grid">
            {categories.map((category) => (
              <button
                className={selectedCategories.includes(category) ? "chip active" : "chip"}
                key={category}
                onClick={() => toggleCategory(category)}
                type="button"
              >
                <TooltipTerm text={categoryLabels[category]} explanation={categoryHelp[category]} compact />
              </button>
            ))}
          </div>
          <button className="run-button" disabled={busy || selectedCategories.length === 0} onClick={startRun} type="button">
            {busy ? <RefreshCw size={18} /> : <Play size={18} />}
            Test prompt
          </button>
        </aside>

        <section className="map-panel" id="metrics">
          <div className="panel-heading">
            <div>
              <h2><TooltipTerm text="Stability map" explanation="A chart showing whether changed prompts produced similar or different answers." /></h2>
              <p>
                <TooltipTerm text="Distance" explanation="A bigger distance means the answer changed more from the original answer." /> from the original answer, drawn as a simple 2D chart.
              </p>
            </div>
            <label className="filter">
              <Filter size={16} />
              <select value={categoryFilter} onChange={(event) => setCategoryFilter(event.target.value as Category | "all")}>
                <option value="all">All categories</option>
                {categories.map((category) => (
                  <option key={category} value={category}>
                    {category}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <Scatter points={filteredPoints} variants={variantById} selectedId={selectedId} onSelect={setSelectedId} />
          <div className="metrics-strip">
            <Metric label="Meaning check" value={run?.metrics?.embedding_device ?? "-"} help="The method used to compare the meaning of two AI answers." />
            <Metric label="Map layout" value={run?.metrics?.projection_method ?? "-"} help="How the app turns answer comparisons into a flat chart." />
            <Metric label="Answer groups" value={run?.metrics?.clustering_method ?? "-"} help="How the app groups answers that look similar." />
            <Metric label="Errors" value={String(run?.metrics?.api_error_count ?? 0)} />
          </div>
        </section>

        <aside className="details" id="details">
          <div className="panel-heading compact">
            <div>
              <h2>Selected test</h2>
              <p>{selectedVariant?.category === "base" ? "original" : selectedVariant?.category ? categoryLabels[selectedVariant.category] : "none"}</p>
            </div>
            {run && (
              <a className="icon-link" href={`${API_BASE}/api/runs/${run.id}/export`} id="export">
                <Download size={17} />
              </a>
            )}
          </div>
          {runError && <div className="run-error">{runError}</div>}
          {selectedVariant && (
            <>
              <div className="variant-meta">
                <span>{selectedVariant.id}</span>
                <span>{selectedVariant.transform}</span>
              </div>
              <h3>Prompt</h3>
              <pre>{selectedVariant.prompt}</pre>
              <h3>Output</h3>
              <pre>{outputText}</pre>
              <h3><TooltipTerm text="Difference from original" explanation="Highlighted words are words that did not appear in the original answer." /></h3>
              <Diff base={baseResponse?.output ?? ""} current={selectedResponse?.output ?? ""} />
              <PointStats point={selectedVariant ? pointById.get(selectedVariant.id) : undefined} />
            </>
          )}
        </aside>
      </section>

      <section className="bottom-grid">
        <div className="summary-panel">
          <h2>Which changes affected the answer most</h2>
          {(run?.metrics?.category_summaries ?? []).length === 0 && <p className="empty-summary">Run a test to compare prompt change types.</p>}
          {(run?.metrics?.category_summaries ?? []).map((summary) => (
            <div className="bar-row" key={summary.category}>
              <span>{categoryLabels[summary.category as Category] ?? summary.category}</span>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${Math.min(100, summary.avg_distance * 100)}%` }} />
              </div>
              <strong>{summary.avg_distance.toFixed(3)}</strong>
            </div>
          ))}
        </div>
        <div className="summary-panel">
          <h2>Biggest answer changes</h2>
          {topShifts.length === 0 && <p className="empty-summary">The largest changes will appear here after a run.</p>}
          {topShifts.map((point) => {
            const variant = variantById.get(point.variant_id);
            return (
              <button className="shift-row" key={point.variant_id} onClick={() => setSelectedId(point.variant_id)} type="button">
                <Activity size={15} />
                <span>{variant?.category && variant.category !== "base" ? categoryLabels[variant.category] : variant?.category}</span>
                <strong>{point.semantic_distance.toFixed(3)}</strong>
              </button>
            );
          })}
        </div>
        <div className="summary-panel training-panel">
          <div className="training-heading">
            <div>
              <h2><TooltipTerm text="GPU training" explanation="Trains a small PyTorch model and uses CUDA automatically when this computer has a compatible GPU." /></h2>
              <p>Learns which prompt changes are likely to cause big answer changes.</p>
            </div>
            <Brain size={22} />
          </div>
          <button className="train-button" disabled={trainingBusy} onClick={trainRiskModel} type="button">
            {trainingBusy || training?.status === "running" ? <RefreshCw size={17} /> : <Zap size={17} />}
            Train Risk Model
          </button>
          {training?.status === "failed" && <p className="training-error">{training.error}</p>}
          <div className="training-grid">
            <Metric label="Device" value={training?.device ?? "-"} help="The processor used for training: cuda means GPU, cpu means processor only." />
            <Metric label="GPU" value={training?.cuda_device_name ?? "-"} help="The GPU name reported by PyTorch when CUDA is available." />
            <Metric label="Examples" value={String(training?.dataset_size ?? 0)} help="Saved prompt variation results used as training examples." />
            <Metric label="Epochs" value={String(training?.epochs ?? 0)} help="How many passes the model made over the training data." />
            <Metric label="Loss" value={training?.final_loss == null ? "-" : training.final_loss.toFixed(4)} help="Lower is better; this is the model's final training error." />
            <Metric label="Accuracy" value={training?.validation_accuracy == null ? "-" : `${Math.round(training.validation_accuracy * 100)}%`} help="How often the model predicted high-risk prompt changes on held-out examples." />
            <Metric label="Time" value={training?.training_ms == null ? "-" : `${(training.training_ms / 1000).toFixed(1)}s`} />
            <Metric label="Status" value={training?.status ?? "idle"} />
          </div>
        </div>
      </section>
    </main>
  );
}

function Scatter({
  points,
  variants,
  selectedId,
  onSelect,
}: {
  points: Point[];
  variants: Map<string, Variant>;
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const bounds = useMemo(() => {
    const xs = points.map((point) => point.x);
    const ys = points.map((point) => point.y);
    return {
      minX: Math.min(...xs, -1),
      maxX: Math.max(...xs, 1),
      minY: Math.min(...ys, -1),
      maxY: Math.max(...ys, 1),
    };
  }, [points]);
  const width = 760;
  const height = 470;
  const pad = 34;

  function scaleX(x: number) {
    return pad + ((x - bounds.minX) / Math.max(0.001, bounds.maxX - bounds.minX)) * (width - pad * 2);
  }

  function scaleY(y: number) {
    return height - pad - ((y - bounds.minY) / Math.max(0.001, bounds.maxY - bounds.minY)) * (height - pad * 2);
  }

  if (!points.length) {
    return <div className="empty-map">Run a map to see response clusters.</div>;
  }

  return (
    <svg className="scatter" viewBox={`0 0 ${width} ${height}`} role="img">
      <rect x="0" y="0" width={width} height={height} rx="8" />
      {points.map((point) => {
        const variant = variants.get(point.variant_id);
        const isBase = point.variant_id === "base";
        const isSelected = selectedId === point.variant_id;
        const color = isBase ? "#111827" : colors[Math.abs(point.cluster) % colors.length];
        return (
          <circle
            key={point.variant_id}
            cx={scaleX(point.x)}
            cy={scaleY(point.y)}
            r={isSelected ? 9 : isBase ? 7 : 5 + Math.min(5, point.semantic_distance * 12)}
            fill={color}
            stroke={isSelected ? "#f8fafc" : "rgba(255,255,255,0.75)"}
            strokeWidth={isSelected ? 3 : 1}
            onClick={() => onSelect(point.variant_id)}
          >
            <title>{`${variant?.category}: ${variant?.transform}`}</title>
          </circle>
        );
      })}
    </svg>
  );
}

function Metric({ label, value, help }: { label: string; value: string; help?: string }) {
  return (
    <div className="metric">
      <span>{help ? <TooltipTerm text={label} explanation={help} compact /> : label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PointStats({ point }: { point?: Point }) {
  if (!point) return null;
  return (
    <div className="stats-grid">
      <Metric label="Distance" value={point.semantic_distance.toFixed(3)} />
      <Metric label="Answer group" value={String(point.cluster)} help="The group this answer belongs to based on similarity." />
      <Metric label="Tokens" value={String(point.token_count)} />
      <Metric label="Variety" value={point.entropy.toFixed(2)} help="A rough measure of how varied the words in the answer are." />
    </div>
  );
}

function TooltipTerm({ text, explanation, compact = false }: { text: string; explanation: string; compact?: boolean }) {
  return (
    <span className={compact ? "tooltip-term compact" : "tooltip-term"} tabIndex={0}>
      <span>{text}</span>
      <Info className="tooltip-icon" size={compact ? 11 : 13} aria-hidden="true" />
      <span className="tooltip-bubble" role="tooltip">{explanation}</span>
    </span>
  );
}

function Diff({ base, current }: { base: string; current: string }) {
  if (!base || !current) return <pre>No diff available.</pre>;
  const baseWords = new Set(base.toLowerCase().split(/\s+/));
  const words = current.split(/\s+/);
  return (
    <div className="diff">
      {words.map((word, index) => (
        <span className={baseWords.has(word.toLowerCase()) ? "same" : "added"} key={`${word}-${index}`}>
          {word}{" "}
        </span>
      ))}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
