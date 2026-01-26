"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { format } from "date-fns";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { MetricInput } from "@/components/metrics/metric-input";
import type { MetricData, EntryData } from "@/types";

interface CheckInFormProps {
  experiment: {
    id: string;
    title: string;
  };
  metrics: MetricData[];
  existingEntry: EntryData | null;
}

const MOODS = [
  { value: "great", label: "Great" },
  { value: "good", label: "Good" },
  { value: "okay", label: "Okay" },
  { value: "low", label: "Low" },
  { value: "rough", label: "Rough" },
];

export function CheckInForm({
  experiment,
  metrics,
  existingEntry,
}: CheckInFormProps) {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);

  // Parse existing values if editing
  const existingValues = existingEntry?.values
    ? JSON.parse(existingEntry.values)
    : {};

  const [values, setValues] = useState<Record<string, unknown>>(existingValues);
  const [notes, setNotes] = useState(existingEntry?.notes || "");
  const [mood, setMood] = useState(existingEntry?.mood || "");

  const today = new Date();

  const updateValue = (metricId: string, value: unknown) => {
    setValues((prev) => ({ ...prev, [metricId]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      const response = await fetch(
        `/api/experiments/${experiment.id}/entries`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            date: today.toISOString(),
            values,
            notes: notes || undefined,
            mood: mood || undefined,
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to save check-in");
      }

      router.push(`/experiments/${experiment.id}`);
      router.refresh();
    } catch (error) {
      console.error("Error saving check-in:", error);
      alert("Failed to save check-in. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>{format(today, "EEEE, MMMM d, yyyy")}</span>
            {existingEntry && (
              <span className="text-sm font-normal text-muted-foreground">
                Updating existing entry
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Metrics */}
          {metrics.map((metric) => (
            <div key={metric.id} className="space-y-2">
              <Label className="text-base">
                {metric.name}
                {metric.description && (
                  <span className="text-sm text-muted-foreground ml-2">
                    ({metric.description})
                  </span>
                )}
              </Label>
              <MetricInput
                metric={metric}
                value={values[metric.id]}
                onChange={(value) => updateValue(metric.id, value)}
              />
            </div>
          ))}

          {metrics.length === 0 && (
            <p className="text-muted-foreground text-center py-4">
              No metrics defined for this experiment.
            </p>
          )}
        </CardContent>
      </Card>

      {/* Optional fields */}
      <Card>
        <CardHeader>
          <CardTitle>Reflection (Optional)</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="mood">How are you feeling?</Label>
            <Select value={mood} onValueChange={setMood}>
              <SelectTrigger>
                <SelectValue placeholder="Select mood" />
              </SelectTrigger>
              <SelectContent>
                {MOODS.map((m) => (
                  <SelectItem key={m.value} value={m.value}>
                    {m.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="notes">Notes & Observations</Label>
            <Textarea
              id="notes"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Any thoughts, observations, or learnings from today?"
              rows={4}
            />
          </div>
        </CardContent>
      </Card>

      {/* Actions */}
      <div className="flex gap-4">
        <Button type="submit" disabled={isLoading} className="flex-1">
          {isLoading
            ? "Saving..."
            : existingEntry
            ? "Update Check-in"
            : "Save Check-in"}
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
