"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Card, CardContent } from "@/components/ui/card";
import { METRIC_TYPES, type MetricType, type CreateMetricInput } from "@/types";

interface MetricBuilderProps {
  metrics: CreateMetricInput[];
  onChange: (metrics: CreateMetricInput[]) => void;
}

const METRIC_TYPE_LABELS: Record<MetricType, string> = {
  boolean: "Yes/No",
  number: "Number",
  scale: "Scale (1-10)",
  text: "Text",
  duration: "Duration",
  select: "Single Choice",
  "multi-select": "Multiple Choice",
};

export function MetricBuilder({ metrics, onChange }: MetricBuilderProps) {
  const addMetric = () => {
    onChange([
      ...metrics,
      {
        name: "",
        type: "boolean",
        isPrimary: metrics.length === 0,
      },
    ]);
  };

  const updateMetric = (index: number, updates: Partial<CreateMetricInput>) => {
    const newMetrics = [...metrics];
    newMetrics[index] = { ...newMetrics[index], ...updates };
    onChange(newMetrics);
  };

  const removeMetric = (index: number) => {
    const newMetrics = metrics.filter((_, i) => i !== index);
    // If we removed the primary, make the first one primary
    if (newMetrics.length > 0 && !newMetrics.some((m) => m.isPrimary)) {
      newMetrics[0].isPrimary = true;
    }
    onChange(newMetrics);
  };

  const setPrimary = (index: number) => {
    const newMetrics = metrics.map((m, i) => ({
      ...m,
      isPrimary: i === index,
    }));
    onChange(newMetrics);
  };

  return (
    <div className="space-y-4">
      {metrics.map((metric, index) => (
        <Card key={index}>
          <CardContent className="pt-4 space-y-4">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Metric Name</Label>
                  <Input
                    value={metric.name}
                    onChange={(e) =>
                      updateMetric(index, { name: e.target.value })
                    }
                    placeholder="e.g., Completed, Energy Level"
                  />
                </div>

                <div className="space-y-2">
                  <Label>Type</Label>
                  <Select
                    value={metric.type}
                    onValueChange={(value) =>
                      updateMetric(index, {
                        type: value as MetricType,
                        config: getDefaultConfig(value as MetricType),
                      })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {METRIC_TYPES.map((t) => (
                        <SelectItem key={t} value={t}>
                          {METRIC_TYPE_LABELS[t]}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => removeMetric(index)}
                className="text-destructive hover:text-destructive"
              >
                Remove
              </Button>
            </div>

            {/* Type-specific config */}
            {metric.type === "scale" && (
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Min Value</Label>
                  <Input
                    type="number"
                    value={
                      (metric.config as { min?: number; max?: number })?.min ?? 1
                    }
                    onChange={(e) =>
                      updateMetric(index, {
                        config: {
                          ...(metric.config as object),
                          min: parseInt(e.target.value),
                        },
                      })
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label>Max Value</Label>
                  <Input
                    type="number"
                    value={
                      (metric.config as { min?: number; max?: number })?.max ??
                      10
                    }
                    onChange={(e) =>
                      updateMetric(index, {
                        config: {
                          ...(metric.config as object),
                          max: parseInt(e.target.value),
                        },
                      })
                    }
                  />
                </div>
              </div>
            )}

            {metric.type === "number" && (
              <div className="space-y-2">
                <Label>Unit (Optional)</Label>
                <Input
                  value={
                    (metric.config as { unit?: string })?.unit ?? ""
                  }
                  onChange={(e) =>
                    updateMetric(index, {
                      config: {
                        ...(metric.config as object),
                        unit: e.target.value,
                      },
                    })
                  }
                  placeholder="e.g., pages, minutes, cups"
                />
              </div>
            )}

            {metric.type === "duration" && (
              <div className="space-y-2">
                <Label>Duration Unit</Label>
                <Select
                  value={
                    (metric.config as { unit?: string })?.unit ?? "minutes"
                  }
                  onValueChange={(value) =>
                    updateMetric(index, {
                      config: { unit: value },
                    })
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="seconds">Seconds</SelectItem>
                    <SelectItem value="minutes">Minutes</SelectItem>
                    <SelectItem value="hours">Hours</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )}

            {(metric.type === "select" || metric.type === "multi-select") && (
              <div className="space-y-2">
                <Label>Options (comma-separated)</Label>
                <Input
                  value={
                    (metric.config as { options?: string[] })?.options?.join(
                      ", "
                    ) ?? ""
                  }
                  onChange={(e) =>
                    updateMetric(index, {
                      config: {
                        options: e.target.value
                          .split(",")
                          .map((s) => s.trim())
                          .filter(Boolean),
                      },
                    })
                  }
                  placeholder="e.g., Happy, Neutral, Sad"
                />
              </div>
            )}

            {/* Primary toggle */}
            <div className="flex items-center space-x-2">
              <Switch
                checked={metric.isPrimary}
                onCheckedChange={() => setPrimary(index)}
              />
              <Label className="text-sm text-muted-foreground">
                Primary metric (shown in charts)
              </Label>
            </div>
          </CardContent>
        </Card>
      ))}

      <Button type="button" variant="outline" onClick={addMetric}>
        + Add Metric
      </Button>

      {metrics.length === 0 && (
        <p className="text-sm text-muted-foreground">
          Add at least one metric to track during your experiment.
        </p>
      )}
    </div>
  );
}

function getDefaultConfig(type: MetricType) {
  switch (type) {
    case "scale":
      return { min: 1, max: 10 };
    case "duration":
      return { unit: "minutes" };
    case "select":
    case "multi-select":
      return { options: [] };
    default:
      return null;
  }
}
