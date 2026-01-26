"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MetricBuilder } from "@/components/metrics/metric-builder";
import {
  EXPERIMENT_TYPES,
  FREQUENCY_TYPES,
  type ExperimentType,
  type FrequencyType,
  type CreateMetricInput,
} from "@/types";

interface ExperimentFormProps {
  initialData?: {
    id?: string;
    title?: string;
    hypothesis?: string;
    type?: ExperimentType;
    intervention?: string;
    successCriteria?: string;
    frequency?: FrequencyType;
    startDate?: string;
    endDate?: string;
    tags?: string[];
    metrics?: CreateMetricInput[];
  };
}

export function ExperimentForm({ initialData }: ExperimentFormProps) {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);

  const [title, setTitle] = useState(initialData?.title || "");
  const [hypothesis, setHypothesis] = useState(initialData?.hypothesis || "");
  const [type, setType] = useState<ExperimentType>(
    initialData?.type || "habit"
  );
  const [intervention, setIntervention] = useState(
    initialData?.intervention || ""
  );
  const [successCriteria, setSuccessCriteria] = useState(
    initialData?.successCriteria || ""
  );
  const [frequency, setFrequency] = useState<FrequencyType>(
    initialData?.frequency || "daily"
  );
  const [startDate, setStartDate] = useState(initialData?.startDate || "");
  const [endDate, setEndDate] = useState(initialData?.endDate || "");
  const [metrics, setMetrics] = useState<CreateMetricInput[]>(
    initialData?.metrics || []
  );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      const payload = {
        title,
        hypothesis,
        type,
        intervention: intervention || undefined,
        successCriteria: successCriteria || undefined,
        frequency,
        startDate: startDate || undefined,
        endDate: endDate || undefined,
        metrics,
      };

      const url = initialData?.id
        ? `/api/experiments/${initialData.id}`
        : "/api/experiments";

      const response = await fetch(url, {
        method: initialData?.id ? "PUT" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error("Failed to save experiment");
      }

      const experiment = await response.json();
      router.push(`/experiments/${experiment.id}`);
      router.refresh();
    } catch (error) {
      console.error("Error saving experiment:", error);
      alert("Failed to save experiment. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      {/* Basic Info */}
      <Card>
        <CardHeader>
          <CardTitle>Experiment Details</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="title">Title</Label>
            <Input
              id="title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g., Morning Pages Practice"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="hypothesis">Hypothesis</Label>
            <Textarea
              id="hypothesis"
              value={hypothesis}
              onChange={(e) => setHypothesis(e.target.value)}
              placeholder="If I do X, then Y will happen because Z..."
              required
              rows={3}
            />
            <p className="text-sm text-muted-foreground">
              What do you expect to happen? Be specific.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="type">Experiment Type</Label>
              <Select
                value={type}
                onValueChange={(value) => setType(value as ExperimentType)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select type" />
                </SelectTrigger>
                <SelectContent>
                  {EXPERIMENT_TYPES.map((t) => (
                    <SelectItem key={t} value={t}>
                      {t.charAt(0).toUpperCase() + t.slice(1)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="frequency">Check-in Frequency</Label>
              <Select
                value={frequency}
                onValueChange={(value) =>
                  setFrequency(value as FrequencyType)
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select frequency" />
                </SelectTrigger>
                <SelectContent>
                  {FREQUENCY_TYPES.map((f) => (
                    <SelectItem key={f} value={f}>
                      {f.charAt(0).toUpperCase() + f.slice(1)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="intervention">Intervention (Optional)</Label>
            <Textarea
              id="intervention"
              value={intervention}
              onChange={(e) => setIntervention(e.target.value)}
              placeholder="What specific action or change are you testing?"
              rows={2}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="successCriteria">Success Criteria (Optional)</Label>
            <Input
              id="successCriteria"
              value={successCriteria}
              onChange={(e) => setSuccessCriteria(e.target.value)}
              placeholder="e.g., Complete 80% of days, average score >= 7"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="startDate">Start Date</Label>
              <Input
                id="startDate"
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="endDate">End Date</Label>
              <Input
                id="endDate"
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Metrics */}
      <Card>
        <CardHeader>
          <CardTitle>Metrics</CardTitle>
          <p className="text-sm text-muted-foreground">
            What will you measure during this experiment?
          </p>
        </CardHeader>
        <CardContent>
          <MetricBuilder metrics={metrics} onChange={setMetrics} />
        </CardContent>
      </Card>

      {/* Actions */}
      <div className="flex gap-4">
        <Button type="submit" disabled={isLoading}>
          {isLoading
            ? "Saving..."
            : initialData?.id
            ? "Update Experiment"
            : "Create Experiment"}
        </Button>
        <Button
          type="button"
          variant="outline"
          onClick={() => router.back()}
          disabled={isLoading}
        >
          Cancel
        </Button>
      </div>
    </form>
  );
}
