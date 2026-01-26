"use client";

import Link from "next/link";
import { format, differenceInDays } from "date-fns";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  EXPERIMENT_TYPE_COLORS,
  EXPERIMENT_STATUS_COLORS,
  type ExperimentWithMetrics,
} from "@/types";

interface ExperimentCardProps {
  experiment: ExperimentWithMetrics;
}

export function ExperimentCard({ experiment }: ExperimentCardProps) {
  const typeColor =
    EXPERIMENT_TYPE_COLORS[
      experiment.type as keyof typeof EXPERIMENT_TYPE_COLORS
    ] || "bg-slate-500";
  const statusColor =
    EXPERIMENT_STATUS_COLORS[
      experiment.status as keyof typeof EXPERIMENT_STATUS_COLORS
    ] || "bg-slate-400";

  // Calculate progress
  const totalDays =
    experiment.startDate && experiment.endDate
      ? differenceInDays(
          new Date(experiment.endDate),
          new Date(experiment.startDate)
        ) + 1
      : 0;

  const daysElapsed = experiment.startDate
    ? Math.min(
        differenceInDays(new Date(), new Date(experiment.startDate)) + 1,
        totalDays
      )
    : 0;

  const entriesCount = experiment._count?.entries || 0;
  const completionRate =
    daysElapsed > 0 ? Math.round((entriesCount / daysElapsed) * 100) : 0;

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${typeColor}`} />
              <CardTitle className="text-lg">{experiment.title}</CardTitle>
            </div>
            <CardDescription className="line-clamp-2">
              {experiment.hypothesis}
            </CardDescription>
          </div>
          <Badge variant="secondary" className={`${statusColor} text-white`}>
            {experiment.status}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Progress bar */}
        {totalDays > 0 && (
          <div className="space-y-1">
            <div className="flex justify-between text-sm text-muted-foreground">
              <span>
                Day {daysElapsed} of {totalDays}
              </span>
              <span>{completionRate}% complete</span>
            </div>
            <div className="h-2 bg-secondary rounded-full overflow-hidden">
              <div
                className="h-full bg-primary transition-all"
                style={{ width: `${Math.min(completionRate, 100)}%` }}
              />
            </div>
          </div>
        )}

        {/* Metrics preview */}
        {experiment.metrics.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {experiment.metrics.slice(0, 3).map((metric) => (
              <Badge key={metric.id} variant="outline" className="text-xs">
                {metric.name}
              </Badge>
            ))}
            {experiment.metrics.length > 3 && (
              <Badge variant="outline" className="text-xs">
                +{experiment.metrics.length - 3} more
              </Badge>
            )}
          </div>
        )}

        {/* Stats */}
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>{entriesCount} entries</span>
          {experiment.startDate && (
            <span>
              Started {format(new Date(experiment.startDate), "MMM d, yyyy")}
            </span>
          )}
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          {experiment.status === "active" && (
            <Button asChild size="sm" className="flex-1">
              <Link href={`/experiments/${experiment.id}/checkin`}>
                Check In
              </Link>
            </Button>
          )}
          <Button asChild variant="outline" size="sm" className="flex-1">
            <Link href={`/experiments/${experiment.id}`}>View Details</Link>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
