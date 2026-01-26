import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { getOrCreateOwner } from "@/lib/auth";

// GET /api/experiments/[id] - Get a single experiment
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const user = await getOrCreateOwner();

    const experiment = await prisma.experiment.findFirst({
      where: {
        id,
        userId: user.id,
      },
      include: {
        metrics: {
          orderBy: { order: "asc" },
        },
        entries: {
          orderBy: { date: "desc" },
        },
      },
    });

    if (!experiment) {
      return NextResponse.json(
        { error: "Experiment not found" },
        { status: 404 }
      );
    }

    return NextResponse.json(experiment);
  } catch (error) {
    console.error("Error fetching experiment:", error);
    return NextResponse.json(
      { error: "Failed to fetch experiment" },
      { status: 500 }
    );
  }
}

// PUT /api/experiments/[id] - Update an experiment
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const user = await getOrCreateOwner();
    const body = await request.json();

    // Check ownership
    const existing = await prisma.experiment.findFirst({
      where: { id, userId: user.id },
    });

    if (!existing) {
      return NextResponse.json(
        { error: "Experiment not found" },
        { status: 404 }
      );
    }

    const {
      title,
      hypothesis,
      type,
      intervention,
      successCriteria,
      status,
      frequency,
      customFrequency,
      startDate,
      endDate,
      actualEndDate,
      conclusion,
      tags,
    } = body;

    const experiment = await prisma.experiment.update({
      where: { id },
      data: {
        ...(title && { title }),
        ...(hypothesis && { hypothesis }),
        ...(type && { type }),
        ...(intervention !== undefined && { intervention }),
        ...(successCriteria !== undefined && { successCriteria }),
        ...(status && { status }),
        ...(frequency && { frequency }),
        ...(customFrequency !== undefined && {
          customFrequency: customFrequency
            ? JSON.stringify(customFrequency)
            : null,
        }),
        ...(startDate !== undefined && {
          startDate: startDate ? new Date(startDate) : null,
        }),
        ...(endDate !== undefined && {
          endDate: endDate ? new Date(endDate) : null,
        }),
        ...(actualEndDate !== undefined && {
          actualEndDate: actualEndDate ? new Date(actualEndDate) : null,
        }),
        ...(conclusion !== undefined && { conclusion }),
        ...(tags !== undefined && {
          tags: tags ? JSON.stringify(tags) : null,
        }),
      },
      include: {
        metrics: {
          orderBy: { order: "asc" },
        },
      },
    });

    return NextResponse.json(experiment);
  } catch (error) {
    console.error("Error updating experiment:", error);
    return NextResponse.json(
      { error: "Failed to update experiment" },
      { status: 500 }
    );
  }
}

// DELETE /api/experiments/[id] - Delete an experiment
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const user = await getOrCreateOwner();

    // Check ownership
    const existing = await prisma.experiment.findFirst({
      where: { id, userId: user.id },
    });

    if (!existing) {
      return NextResponse.json(
        { error: "Experiment not found" },
        { status: 404 }
      );
    }

    await prisma.experiment.delete({
      where: { id },
    });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error deleting experiment:", error);
    return NextResponse.json(
      { error: "Failed to delete experiment" },
      { status: 500 }
    );
  }
}
