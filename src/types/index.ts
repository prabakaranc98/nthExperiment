// Experiment types
export const EXPERIMENT_TYPES = [
  "habit",
  "wellbeing",
  "creative",
  "social",
  "data",
  "learning",
  "custom",
] as const;

export type ExperimentType = (typeof EXPERIMENT_TYPES)[number];

// Experiment status
export const EXPERIMENT_STATUSES = [
  "draft",
  "active",
  "paused",
  "completed",
  "archived",
] as const;

export type ExperimentStatus = (typeof EXPERIMENT_STATUSES)[number];

// Frequency types
export const FREQUENCY_TYPES = [
  "daily",
  "weekdays",
  "weekly",
  "custom",
] as const;

export type FrequencyType = (typeof FREQUENCY_TYPES)[number];

// Metric types
export const METRIC_TYPES = [
  "boolean",
  "number",
  "scale",
  "text",
  "duration",
  "select",
  "multi-select",
] as const;

export type MetricType = (typeof METRIC_TYPES)[number];

// Metric configuration types
export interface ScaleConfig {
  min: number;
  max: number;
  labels?: Record<number, string>;
}

export interface NumberConfig {
  unit?: string;
  min?: number;
  max?: number;
}

export interface SelectConfig {
  options: string[];
}

export interface DurationConfig {
  unit: "seconds" | "minutes" | "hours";
}

export type MetricConfig =
  | ScaleConfig
  | NumberConfig
  | SelectConfig
  | DurationConfig
  | null;

// Type colors for UI
export const EXPERIMENT_TYPE_COLORS: Record<ExperimentType, string> = {
  habit: "bg-emerald-500",
  wellbeing: "bg-rose-500",
  creative: "bg-violet-500",
  social: "bg-blue-500",
  data: "bg-amber-500",
  learning: "bg-cyan-500",
  custom: "bg-slate-500",
};

// Status colors for UI
export const EXPERIMENT_STATUS_COLORS: Record<ExperimentStatus, string> = {
  draft: "bg-slate-400",
  active: "bg-emerald-500",
  paused: "bg-amber-500",
  completed: "bg-blue-500",
  archived: "bg-slate-300",
};

// Form types
export interface CreateExperimentInput {
  title: string;
  hypothesis: string;
  type: ExperimentType;
  intervention?: string;
  successCriteria?: string;
  frequency: FrequencyType;
  customFrequency?: Record<string, unknown>;
  startDate?: Date;
  endDate?: Date;
  tags?: string[];
}

export interface CreateMetricInput {
  name: string;
  description?: string;
  type: MetricType;
  config?: MetricConfig;
  isPrimary?: boolean;
  order?: number;
}

export interface CreateEntryInput {
  experimentId: string;
  date: Date;
  values: Record<string, unknown>;
  notes?: string;
  mood?: string;
}

// API response types
export interface ExperimentWithMetrics {
  id: string;
  userId: string;
  title: string;
  hypothesis: string;
  type: string;
  intervention: string | null;
  successCriteria: string | null;
  status: string;
  frequency: string;
  customFrequency: string | null;
  startDate: Date | null;
  endDate: Date | null;
  actualEndDate: Date | null;
  conclusion: string | null;
  tags: string | null;
  createdAt: Date;
  updatedAt: Date;
  metrics: MetricData[];
  entries?: EntryData[];
  _count?: {
    entries: number;
  };
}

export interface MetricData {
  id: string;
  experimentId: string;
  name: string;
  description: string | null;
  type: string;
  config: string | null;
  isPrimary: boolean;
  order: number;
  createdAt: Date;
}

export interface EntryData {
  id: string;
  experimentId: string;
  date: Date;
  values: string;
  notes: string | null;
  mood: string | null;
  createdAt: Date;
  updatedAt: Date;
}

// Computed stats
export interface ExperimentStats {
  totalDays: number;
  entriesCount: number;
  completionRate: number;
  currentStreak: number;
  averages: Record<string, number>;
}
