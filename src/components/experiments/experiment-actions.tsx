"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";

interface ExperimentActionsProps {
  experiment: {
    id: string;
    status: string;
    title: string;
  };
}

export function ExperimentActions({ experiment }: ExperimentActionsProps) {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [showCompleteDialog, setShowCompleteDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [conclusion, setConclusion] = useState("");

  const updateStatus = async (status: string, additionalData?: object) => {
    setIsLoading(true);
    try {
      const response = await fetch(`/api/experiments/${experiment.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status, ...additionalData }),
      });

      if (!response.ok) throw new Error("Failed to update");
      router.refresh();
    } catch (error) {
      console.error("Error updating experiment:", error);
      alert("Failed to update experiment");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`/api/experiments/${experiment.id}`, {
        method: "DELETE",
      });

      if (!response.ok) throw new Error("Failed to delete");
      router.push("/experiments");
      router.refresh();
    } catch (error) {
      console.error("Error deleting experiment:", error);
      alert("Failed to delete experiment");
    } finally {
      setIsLoading(false);
      setShowDeleteDialog(false);
    }
  };

  const handleComplete = async () => {
    await updateStatus("completed", {
      conclusion,
      actualEndDate: new Date().toISOString(),
    });
    setShowCompleteDialog(false);
  };

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" disabled={isLoading}>
            Actions
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          {experiment.status === "draft" && (
            <DropdownMenuItem onClick={() => updateStatus("active")}>
              Start Experiment
            </DropdownMenuItem>
          )}
          {experiment.status === "active" && (
            <>
              <DropdownMenuItem onClick={() => updateStatus("paused")}>
                Pause Experiment
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setShowCompleteDialog(true)}>
                Complete Experiment
              </DropdownMenuItem>
            </>
          )}
          {experiment.status === "paused" && (
            <>
              <DropdownMenuItem onClick={() => updateStatus("active")}>
                Resume Experiment
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setShowCompleteDialog(true)}>
                Complete Experiment
              </DropdownMenuItem>
            </>
          )}
          <DropdownMenuItem
            onClick={() => router.push(`/experiments/${experiment.id}/edit`)}
          >
            Edit Details
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            className="text-destructive"
            onClick={() => setShowDeleteDialog(true)}
          >
            Delete Experiment
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      {/* Complete Dialog */}
      <Dialog open={showCompleteDialog} onOpenChange={setShowCompleteDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Complete Experiment</DialogTitle>
            <DialogDescription>
              Add a conclusion to summarize what you learned from this
              experiment.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="conclusion">Conclusion (Optional)</Label>
              <Textarea
                id="conclusion"
                value={conclusion}
                onChange={(e) => setConclusion(e.target.value)}
                placeholder="What did you learn? Would you continue this practice?"
                rows={4}
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowCompleteDialog(false)}
            >
              Cancel
            </Button>
            <Button onClick={handleComplete} disabled={isLoading}>
              {isLoading ? "Completing..." : "Complete Experiment"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Experiment</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete &ldquo;{experiment.title}&rdquo;?
              This action cannot be undone and all entries will be lost.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowDeleteDialog(false)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDelete}
              disabled={isLoading}
            >
              {isLoading ? "Deleting..." : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
