import Link from "next/link";
import { format } from "date-fns";
import { prisma } from "@/lib/db";
import { getOrCreateOwner } from "@/lib/auth";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { EXPERIMENT_TYPE_COLORS } from "@/types";
import type { ExperimentWithMetrics } from "@/types";

async function getDashboardData() {
  const user = await getOrCreateOwner();

  // Get active experiments with today's entry status
  const activeExperiments = await prisma.experiment.findMany({
    where: {
      userId: user.id,
      status: "active",
    },
    include: {
      metrics: {
        orderBy: { order: "asc" },
      },
      _count: {
        select: { entries: true },
      },
    },
    orderBy: { updatedAt: "desc" },
  });

  // Check which experiments have entries for today
  const today = new Date();
  today.setUTCHours(0, 0, 0, 0);

  const todayEntries = await prisma.entry.findMany({
    where: {
      experimentId: {
        in: activeExperiments.map((e) => e.id),
      },
      date: today,
    },
  });

  const todayEntryMap = new Set(todayEntries.map((e) => e.experimentId));

  // Get overall stats
  const totalExperiments = await prisma.experiment.count({
    where: { userId: user.id },
  });

  const completedExperiments = await prisma.experiment.count({
    where: { userId: user.id, status: "completed" },
  });

  const totalEntries = await prisma.entry.count({
    where: {
      experiment: {
        userId: user.id,
      },
    },
  });

  // Get recent entries
  const recentEntries = await prisma.entry.findMany({
    where: {
      experiment: {
        userId: user.id,
      },
    },
    include: {
      experiment: {
        select: {
          title: true,
          type: true,
        },
      },
    },
    orderBy: { createdAt: "desc" },
    take: 5,
  });

  return {
    activeExperiments: activeExperiments as ExperimentWithMetrics[],
    todayEntryMap,
    stats: {
      totalExperiments,
      activeExperiments: activeExperiments.length,
      completedExperiments,
      totalEntries,
    },
    recentEntries,
  };
}

export default async function DashboardPage() {
  const { activeExperiments, todayEntryMap, stats, recentEntries } =
    await getDashboardData();

  const pendingCheckins = activeExperiments.filter(
    (e) => !todayEntryMap.has(e.id)
  );
  const completedToday = activeExperiments.filter((e) =>
    todayEntryMap.has(e.id)
  );

  return (
    <div className="container py-6 sm:py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl sm:text-3xl font-bold">Dashboard</h1>
        <p className="text-muted-foreground mt-1">
          {format(new Date(), "EEEE, MMMM d, yyyy")}
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-3 sm:gap-4 md:grid-cols-4 mb-6 sm:mb-8">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Active Experiments
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats.activeExperiments}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Experiments
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats.totalExperiments}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Completed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {stats.completedExperiments}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Entries
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats.totalEntries}</div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 sm:gap-8 lg:grid-cols-3">
        {/* Today's Check-ins */}
        <div className="lg:col-span-2 space-y-6">
          {/* Pending */}
          {pendingCheckins.length > 0 && (
            <div>
              <h2 className="text-xl font-semibold mb-4">
                Today&apos;s Check-ins
                <Badge variant="secondary" className="ml-2">
                  {pendingCheckins.length} pending
                </Badge>
              </h2>
              <div className="grid gap-3 sm:gap-4 sm:grid-cols-2">
                {pendingCheckins.map((experiment) => (
                  <Card
                    key={experiment.id}
                    className="border-l-4 border-l-amber-500"
                  >
                    <CardContent className="pt-4">
                      <div className="flex items-start justify-between">
                        <div>
                          <h3 className="font-medium">{experiment.title}</h3>
                          <p className="text-sm text-muted-foreground capitalize">
                            {experiment.type}
                          </p>
                        </div>
                        <Button asChild size="sm">
                          <Link href={`/experiments/${experiment.id}/checkin`}>
                            Check In
                          </Link>
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {/* Completed Today */}
          {completedToday.length > 0 && (
            <div>
              <h2 className="text-xl font-semibold mb-4">
                Completed Today
                <Badge
                  variant="secondary"
                  className="ml-2 bg-emerald-100 text-emerald-700"
                >
                  {completedToday.length} done
                </Badge>
              </h2>
              <div className="grid gap-3 sm:gap-4 sm:grid-cols-2">
                {completedToday.map((experiment) => (
                  <Card
                    key={experiment.id}
                    className="border-l-4 border-l-emerald-500 opacity-75"
                  >
                    <CardContent className="pt-4">
                      <div className="flex items-start justify-between">
                        <div>
                          <h3 className="font-medium">{experiment.title}</h3>
                          <p className="text-sm text-muted-foreground capitalize">
                            {experiment.type}
                          </p>
                        </div>
                        <Button asChild variant="ghost" size="sm">
                          <Link href={`/experiments/${experiment.id}`}>
                            View
                          </Link>
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {/* No active experiments */}
          {activeExperiments.length === 0 && (
            <Card>
              <CardContent className="py-12 text-center">
                <h2 className="text-xl font-semibold mb-2">
                  No active experiments
                </h2>
                <p className="text-muted-foreground mb-4">
                  Start your first experiment to begin tracking your personal
                  growth.
                </p>
                <Button asChild>
                  <Link href="/experiments/new">Create Experiment</Link>
                </Button>
              </CardContent>
            </Card>
          )}

          {/* All done for today */}
          {activeExperiments.length > 0 && pendingCheckins.length === 0 && (
            <Card className="bg-emerald-50 border-emerald-200">
              <CardContent className="py-8 text-center">
                <h2 className="text-xl font-semibold text-emerald-800 mb-2">
                  All caught up!
                </h2>
                <p className="text-emerald-600">
                  You&apos;ve completed all your check-ins for today. Great
                  work!
                </p>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Recent Activity */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Recent Activity</CardTitle>
            </CardHeader>
            <CardContent>
              {recentEntries.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No recent activity
                </p>
              ) : (
                <div className="space-y-3">
                  {recentEntries.map((entry) => {
                    const typeColor =
                      EXPERIMENT_TYPE_COLORS[
                        entry.experiment
                          .type as keyof typeof EXPERIMENT_TYPE_COLORS
                      ] || "bg-slate-500";

                    return (
                      <div
                        key={entry.id}
                        className="flex items-start gap-3 text-sm"
                      >
                        <span
                          className={`w-2 h-2 rounded-full mt-1.5 ${typeColor}`}
                        />
                        <div>
                          <p className="font-medium">
                            {entry.experiment.title}
                          </p>
                          <p className="text-muted-foreground">
                            {format(
                              new Date(entry.createdAt),
                              "MMM d, h:mm a"
                            )}
                          </p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Quick Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button asChild className="w-full" variant="outline">
                <Link href="/experiments/new">New Experiment</Link>
              </Button>
              <Button asChild className="w-full" variant="outline">
                <Link href="/experiments">View All Experiments</Link>
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
