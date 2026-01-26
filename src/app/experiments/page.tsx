import Link from "next/link";
import { prisma } from "@/lib/db";
import { getOrCreateOwner } from "@/lib/auth";
import { ExperimentCard } from "@/components/experiments/experiment-card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { ExperimentWithMetrics } from "@/types";

async function getExperiments(): Promise<ExperimentWithMetrics[]> {
  const user = await getOrCreateOwner();

  const experiments = await prisma.experiment.findMany({
    where: { userId: user.id },
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

  return experiments as ExperimentWithMetrics[];
}

export default async function ExperimentsPage() {
  const experiments = await getExperiments();

  const activeExperiments = experiments.filter((e) => e.status === "active");
  const draftExperiments = experiments.filter((e) => e.status === "draft");
  const completedExperiments = experiments.filter(
    (e) => e.status === "completed" || e.status === "archived"
  );
  const pausedExperiments = experiments.filter((e) => e.status === "paused");

  return (
    <div className="container py-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">Experiments</h1>
          <p className="text-muted-foreground mt-1">
            Design, track, and analyze your personal experiments
          </p>
        </div>
        <Button asChild>
          <Link href="/experiments/new">New Experiment</Link>
        </Button>
      </div>

      {experiments.length === 0 ? (
        <div className="text-center py-12">
          <h2 className="text-xl font-semibold mb-2">No experiments yet</h2>
          <p className="text-muted-foreground mb-4">
            Start your first experiment to begin tracking your personal growth.
          </p>
          <Button asChild>
            <Link href="/experiments/new">Create Your First Experiment</Link>
          </Button>
        </div>
      ) : (
        <Tabs defaultValue="active" className="space-y-6">
          <TabsList>
            <TabsTrigger value="active">
              Active ({activeExperiments.length})
            </TabsTrigger>
            <TabsTrigger value="draft">
              Draft ({draftExperiments.length})
            </TabsTrigger>
            <TabsTrigger value="paused">
              Paused ({pausedExperiments.length})
            </TabsTrigger>
            <TabsTrigger value="completed">
              Completed ({completedExperiments.length})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="active" className="space-y-4">
            {activeExperiments.length === 0 ? (
              <p className="text-muted-foreground py-8 text-center">
                No active experiments. Start one from your drafts!
              </p>
            ) : (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {activeExperiments.map((experiment) => (
                  <ExperimentCard key={experiment.id} experiment={experiment} />
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="draft" className="space-y-4">
            {draftExperiments.length === 0 ? (
              <p className="text-muted-foreground py-8 text-center">
                No draft experiments.
              </p>
            ) : (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {draftExperiments.map((experiment) => (
                  <ExperimentCard key={experiment.id} experiment={experiment} />
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="paused" className="space-y-4">
            {pausedExperiments.length === 0 ? (
              <p className="text-muted-foreground py-8 text-center">
                No paused experiments.
              </p>
            ) : (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {pausedExperiments.map((experiment) => (
                  <ExperimentCard key={experiment.id} experiment={experiment} />
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="completed" className="space-y-4">
            {completedExperiments.length === 0 ? (
              <p className="text-muted-foreground py-8 text-center">
                No completed experiments yet.
              </p>
            ) : (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {completedExperiments.map((experiment) => (
                  <ExperimentCard key={experiment.id} experiment={experiment} />
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}
