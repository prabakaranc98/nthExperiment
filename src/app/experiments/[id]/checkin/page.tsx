import { notFound } from "next/navigation";
import { prisma } from "@/lib/db";
import { getOrCreateOwner } from "@/lib/auth";
import { CheckInForm } from "@/components/experiments/checkin-form";

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

async function getTodayEntry(experimentId: string) {
  const today = new Date();
  today.setUTCHours(0, 0, 0, 0);

  const entry = await prisma.entry.findFirst({
    where: {
      experimentId,
      date: today,
    },
  });

  return entry;
}

export default async function CheckInPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const experiment = await getExperiment(id);

  if (!experiment) {
    notFound();
  }

  if (experiment.status !== "active") {
    return (
      <div className="container py-8 max-w-2xl">
        <div className="text-center py-12">
          <h1 className="text-2xl font-bold mb-2">Experiment Not Active</h1>
          <p className="text-muted-foreground">
            This experiment is currently {experiment.status}. You can only check
            in to active experiments.
          </p>
        </div>
      </div>
    );
  }

  const todayEntry = await getTodayEntry(experiment.id);

  return (
    <div className="container py-8 max-w-2xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">{experiment.title}</h1>
        <p className="text-muted-foreground mt-1">Daily Check-in</p>
      </div>

      <CheckInForm
        experiment={experiment}
        metrics={experiment.metrics}
        existingEntry={todayEntry}
      />
    </div>
  );
}
