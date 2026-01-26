import { ExperimentForm } from "@/components/experiments/experiment-form";

export default function NewExperimentPage() {
  return (
    <div className="container py-8 max-w-3xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">New Experiment</h1>
        <p className="text-muted-foreground mt-1">
          Design a new experiment to test your hypothesis
        </p>
      </div>

      <ExperimentForm />
    </div>
  );
}
