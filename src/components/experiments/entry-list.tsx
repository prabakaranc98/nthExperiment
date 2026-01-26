"use client";

import { format } from "date-fns";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { EntryData, MetricData } from "@/types";

interface EntryListProps {
  entries: EntryData[];
  metrics: MetricData[];
  experimentId: string;
}

export function EntryList({ entries, metrics }: EntryListProps) {
  if (entries.length === 0) {
    return (
      <Card>
        <CardContent className="py-12 text-center">
          <p className="text-muted-foreground">
            No entries yet. Start checking in to track your progress!
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-3">
      {entries.map((entry) => {
        const values = JSON.parse(entry.values);

        return (
          <Card key={entry.id}>
            <CardContent className="py-4">
              <div className="flex items-start justify-between">
                <div className="space-y-2">
                  <div className="flex items-center gap-3">
                    <span className="font-medium">
                      {format(new Date(entry.date), "EEEE, MMM d, yyyy")}
                    </span>
                    {entry.mood && (
                      <Badge variant="secondary">{entry.mood}</Badge>
                    )}
                  </div>

                  {/* Display metric values */}
                  <div className="flex flex-wrap gap-3 text-sm">
                    {metrics.map((metric) => {
                      const value = values[metric.id];
                      if (value === undefined || value === null) return null;

                      return (
                        <div key={metric.id} className="flex items-center gap-1">
                          <span className="text-muted-foreground">
                            {metric.name}:
                          </span>
                          <span className="font-medium">
                            {formatValue(value, metric)}
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  {/* Notes */}
                  {entry.notes && (
                    <p className="text-sm text-muted-foreground mt-2 italic">
                      &ldquo;{entry.notes}&rdquo;
                    </p>
                  )}
                </div>

                <span className="text-xs text-muted-foreground">
                  {format(new Date(entry.createdAt), "h:mm a")}
                </span>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

function formatValue(value: unknown, metric: MetricData): string {
  const config = metric.config ? JSON.parse(metric.config) : {};

  switch (metric.type) {
    case "boolean":
      return value ? "Yes" : "No";
    case "number":
      return `${value}${config.unit ? ` ${config.unit}` : ""}`;
    case "scale":
      return `${value}/${config.max || 10}`;
    case "duration":
      return `${value} ${config.unit || "minutes"}`;
    case "select":
      return String(value);
    case "multi-select":
      return Array.isArray(value) ? value.join(", ") : String(value);
    case "text":
      return String(value).slice(0, 50) + (String(value).length > 50 ? "..." : "");
    default:
      return String(value);
  }
}
