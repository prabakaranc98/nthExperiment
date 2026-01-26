import { notFound } from "next/navigation";
import { prisma } from "@/lib/db";
import { getOrCreateOwner } from "@/lib/auth";
import { ExperimentForm } from "@/components/experiments/experiment-form";
import { format } from "date-fns";

async function getExperiment(id: string) {
  const user = await getOrCreateOwner();

  const experiment = await prisma.experiment.findFirst({
    where: { id, userId: user.id },
    include: {
      metrics: {
        orderBy: { order: "asc" },
      },
    },
  });

  return experiment;
}

export default async function EditExperimentPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const experiment = await getExperiment(id);

  if (!experiment) {
    notFound();
  }

  // Parse tags
  const tags = experiment.tags ? JSON.parse(experiment.tags) : [];

  // Transform metrics for the form
  const metrics = experiment.metrics.map((m) => ({
    name: m.name,
    description: m.description || undefined,
    type: m.type as
      | "boolean"
      | "number"
      | "scale"
      | "text"
      | "duration"
      | "select"
      | "multi-select",
    config: m.config ? JSON.parse(m.config) : undefined,
    isPrimary: m.isPrimary,
    order: m.order,
  }));

  return (
    <div className="container py-8 max-w-3xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Edit Experiment</h1>
        <p className="text-muted-foreground mt-1">{experiment.title}</p>
      </div>

      <ExperimentForm
        initialData={{
          id: experiment.id,
          title: experiment.title,
          hypothesis: experiment.hypothesis,
          type: experiment.type as
            | "habit"
            | "wellbeing"
            | "creative"
            | "social"
            | "data"
            | "learning"
            | "custom",
          intervention: experiment.intervention || undefined,
          successCriteria: experiment.successCriteria || undefined,
          frequency: experiment.frequency as
            | "daily"
            | "weekdays"
            | "weekly"
            | "custom",
          startDate: experiment.startDate
            ? format(new Date(experiment.startDate), "yyyy-MM-dd")
            : undefined,
          endDate: experiment.endDate
            ? format(new Date(experiment.endDate), "yyyy-MM-dd")
            : undefined,
          tags,
          metrics,
        }}
      />
    </div>
  );
}
