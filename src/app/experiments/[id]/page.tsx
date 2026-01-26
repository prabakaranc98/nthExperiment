import { notFound } from "next/navigation";
import Link from "next/link";
import { format, differenceInDays } from "date-fns";
import { prisma } from "@/lib/db";
import { getOrCreateOwner } from "@/lib/auth";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { ProgressChart } from "@/components/charts/progress-chart";
import { EntryList } from "@/components/experiments/entry-list";
import { ExperimentActions } from "@/components/experiments/experiment-actions";
import {
  EXPERIMENT_TYPE_COLORS,
  EXPERIMENT_STATUS_COLORS,
} from "@/types";

async function getExperiment(id: string) {
  const user = await getOrCreateOwner();

  const experiment = await prisma.experiment.findFirst({
    where: { id, userId: user.id },
    include: {
      metrics: {
        orderBy: { order: "asc" },
      },
      entries: {
        orderBy: { date: "desc" },
      },
    },
  });

  return experiment;
}

export default async function ExperimentDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const experiment = await getExperiment(id);

  if (!experiment) {
    notFound();
  }

  const typeColor =
    EXPERIMENT_TYPE_COLORS[
      experiment.type as keyof typeof EXPERIMENT_TYPE_COLORS
    ] || "bg-slate-500";
  const statusColor =
    EXPERIMENT_STATUS_COLORS[
      experiment.status as keyof typeof EXPERIMENT_STATUS_COLORS
    ] || "bg-slate-400";

  // Calculate stats
  const totalDays =
    experiment.startDate && experiment.endDate
      ? differenceInDays(
          new Date(experiment.endDate),
          new Date(experiment.startDate)
        ) + 1
      : 0;

  const daysElapsed = experiment.startDate
    ? Math.max(
        0,
        Math.min(
          differenceInDays(new Date(), new Date(experiment.startDate)) + 1,
          totalDays
        )
      )
    : 0;

  const completionRate =
    daysElapsed > 0
      ? Math.round((experiment.entries.length / daysElapsed) * 100)
      : 0;

  // Parse tags
  const tags = experiment.tags ? JSON.parse(experiment.tags) : [];

  return (
    <div className="container py-8">
      {/* Header */}
      <div className="flex items-start justify-between mb-8">
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <span className={`w-3 h-3 rounded-full ${typeColor}`} />
            <h1 className="text-3xl font-bold">{experiment.title}</h1>
            <Badge className={`${statusColor} text-white`}>
              {experiment.status}
            </Badge>
          </div>
          <p className="text-lg text-muted-foreground max-w-2xl">
            {experiment.hypothesis}
          </p>
          {tags.length > 0 && (
            <div className="flex gap-2 mt-2">
              {tags.map((tag: string) => (
                <Badge key={tag} variant="outline">
                  {tag}
                </Badge>
              ))}
            </div>
          )}
        </div>
        <div className="flex gap-2">
          {experiment.status === "active" && (
            <Button asChild>
              <Link href={`/experiments/${experiment.id}/checkin`}>
                Check In
              </Link>
            </Button>
          )}
          <ExperimentActions experiment={experiment} />
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-4 mb-8">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Progress
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {daysElapsed} / {totalDays || "∞"}
            </div>
            <p className="text-xs text-muted-foreground">days</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Entries
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{experiment.entries.length}</div>
            <p className="text-xs text-muted-foreground">check-ins</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Completion Rate
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{completionRate}%</div>
            <p className="text-xs text-muted-foreground">of days tracked</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Frequency
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold capitalize">
              {experiment.frequency}
            </div>
            <p className="text-xs text-muted-foreground">check-in schedule</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="progress" className="space-y-6">
        <TabsList>
          <TabsTrigger value="progress">Progress</TabsTrigger>
          <TabsTrigger value="entries">
            Entries ({experiment.entries.length})
          </TabsTrigger>
          <TabsTrigger value="details">Details</TabsTrigger>
        </TabsList>

        <TabsContent value="progress" className="space-y-6">
          {experiment.entries.length > 0 ? (
            <Card>
              <CardHeader>
                <CardTitle>Progress Over Time</CardTitle>
              </CardHeader>
              <CardContent>
                <ProgressChart
                  entries={experiment.entries}
                  metrics={experiment.metrics}
                />
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="py-12 text-center">
                <p className="text-muted-foreground mb-4">
                  No entries yet. Start tracking to see your progress!
                </p>
                {experiment.status === "active" && (
                  <Button asChild>
                    <Link href={`/experiments/${experiment.id}/checkin`}>
                      Add First Entry
                    </Link>
                  </Button>
                )}
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="entries">
          <EntryList
            entries={experiment.entries}
            metrics={experiment.metrics}
            experimentId={experiment.id}
          />
        </TabsContent>

        <TabsContent value="details" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Experiment Details</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {experiment.intervention && (
                <div>
                  <h4 className="font-medium mb-1">Intervention</h4>
                  <p className="text-muted-foreground">
                    {experiment.intervention}
                  </p>
                </div>
              )}

              {experiment.successCriteria && (
                <div>
                  <h4 className="font-medium mb-1">Success Criteria</h4>
                  <p className="text-muted-foreground">
                    {experiment.successCriteria}
                  </p>
                </div>
              )}

              <Separator />

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">Start Date:</span>{" "}
                  {experiment.startDate
                    ? format(new Date(experiment.startDate), "MMM d, yyyy")
                    : "Not set"}
                </div>
                <div>
                  <span className="text-muted-foreground">End Date:</span>{" "}
                  {experiment.endDate
                    ? format(new Date(experiment.endDate), "MMM d, yyyy")
                    : "Not set"}
                </div>
                <div>
                  <span className="text-muted-foreground">Created:</span>{" "}
                  {format(new Date(experiment.createdAt), "MMM d, yyyy")}
                </div>
                <div>
                  <span className="text-muted-foreground">Type:</span>{" "}
                  <span className="capitalize">{experiment.type}</span>
                </div>
              </div>

              {experiment.conclusion && (
                <>
                  <Separator />
                  <div>
                    <h4 className="font-medium mb-1">Conclusion</h4>
                    <p className="text-muted-foreground">
                      {experiment.conclusion}
                    </p>
                  </div>
                </>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {experiment.metrics.map((metric) => (
                  <div
                    key={metric.id}
                    className="flex items-center justify-between p-3 border rounded-lg"
                  >
                    <div>
                      <div className="font-medium">
                        {metric.name}
                        {metric.isPrimary && (
                          <Badge variant="secondary" className="ml-2 text-xs">
                            Primary
                          </Badge>
                        )}
                      </div>
                      {metric.description && (
                        <p className="text-sm text-muted-foreground">
                          {metric.description}
                        </p>
                      )}
                    </div>
                    <Badge variant="outline" className="capitalize">
                      {metric.type}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
