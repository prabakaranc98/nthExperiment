"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { format } from "date-fns";
import type { EntryData, MetricData } from "@/types";

interface ProgressChartProps {
  entries: EntryData[];
  metrics: MetricData[];
}

const COLORS = [
  "#6366f1", // indigo
  "#10b981", // emerald
  "#f59e0b", // amber
  "#ef4444", // red
  "#8b5cf6", // violet
  "#06b6d4", // cyan
];

export function ProgressChart({ entries, metrics }: ProgressChartProps) {
  // Get chartable metrics (numeric types only)
  const chartableMetrics = metrics.filter((m) =>
    ["number", "scale", "duration", "boolean"].includes(m.type)
  );

  if (chartableMetrics.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-muted-foreground">
        No numeric metrics to chart. Add number, scale, or duration metrics to
        see progress.
      </div>
    );
  }

  // Sort entries by date ascending for chart
  const sortedEntries = [...entries].sort(
    (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
  );

  // Transform data for Recharts
  const chartData = sortedEntries.map((entry) => {
    const values = JSON.parse(entry.values);
    const dataPoint: Record<string, unknown> = {
      date: format(new Date(entry.date), "MMM d"),
      fullDate: entry.date,
    };

    chartableMetrics.forEach((metric) => {
      let value = values[metric.id];
      // Convert boolean to 1/0 for charting
      if (metric.type === "boolean") {
        value = value ? 1 : 0;
      }
      dataPoint[metric.id] = value;
    });

    return dataPoint;
  });

  // Find the primary metric or use first chartable metric
  const primaryMetric =
    chartableMetrics.find((m) => m.isPrimary) || chartableMetrics[0];

  // Get Y-axis domain based on metric type
  const getYDomain = (metric: MetricData): [number, number] | undefined => {
    if (metric.type === "boolean") return [0, 1];
    if (metric.type === "scale") {
      const config = metric.config ? JSON.parse(metric.config) : {};
      return [config.min || 1, config.max || 10];
    }
    return undefined; // Auto domain
  };

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={chartData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis
            dataKey="date"
            className="text-xs"
            tick={{ fill: "hsl(var(--muted-foreground))" }}
          />
          <YAxis
            domain={getYDomain(primaryMetric)}
            className="text-xs"
            tick={{ fill: "hsl(var(--muted-foreground))" }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "8px",
            }}
            labelStyle={{ color: "hsl(var(--foreground))" }}
          />
          {chartableMetrics.length > 1 && <Legend />}
          {chartableMetrics.map((metric, index) => (
            <Line
              key={metric.id}
              type="monotone"
              dataKey={metric.id}
              name={metric.name}
              stroke={COLORS[index % COLORS.length]}
              strokeWidth={metric.isPrimary ? 3 : 2}
              dot={{ r: metric.isPrimary ? 4 : 3 }}
              activeDot={{ r: 6 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
