import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { getOrCreateOwner } from "@/lib/auth";

// POST /api/experiments/[id]/metrics - Add a metric to an experiment
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const user = await getOrCreateOwner();
    const body = await request.json();

    // Verify ownership
    const experiment = await prisma.experiment.findFirst({
      where: { id, userId: user.id },
      include: { metrics: true },
    });

    if (!experiment) {
      return NextResponse.json(
        { error: "Experiment not found" },
        { status: 404 }
      );
    }

    const { name, description, type, config, isPrimary } = body;

    if (!name || !type) {
      return NextResponse.json(
        { error: "Name and type are required" },
        { status: 400 }
      );
    }

    // If this is marked as primary, unset other primary metrics
    if (isPrimary) {
      await prisma.metric.updateMany({
        where: { experimentId: id },
        data: { isPrimary: false },
      });
    }

    const metric = await prisma.metric.create({
      data: {
        experimentId: id,
        name,
        description,
        type,
        config: config ? JSON.stringify(config) : null,
        isPrimary: isPrimary || experiment.metrics.length === 0,
        order: experiment.metrics.length,
      },
    });

    return NextResponse.json(metric, { status: 201 });
  } catch (error) {
    console.error("Error creating metric:", error);
    return NextResponse.json(
      { error: "Failed to create metric" },
      { status: 500 }
    );
  }
}

// PUT /api/experiments/[id]/metrics - Update metrics order or bulk update
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const user = await getOrCreateOwner();
    const body = await request.json();

    // Verify ownership
    const experiment = await prisma.experiment.findFirst({
      where: { id, userId: user.id },
    });

    if (!experiment) {
      return NextResponse.json(
        { error: "Experiment not found" },
        { status: 404 }
      );
    }

    const { metrics } = body;

    if (!Array.isArray(metrics)) {
      return NextResponse.json(
        { error: "Metrics array is required" },
        { status: 400 }
      );
    }

    // Update each metric
    await Promise.all(
      metrics.map(
        (
          m: {
            id: string;
            name?: string;
            description?: string;
            type?: string;
            config?: Record<string, unknown>;
            isPrimary?: boolean;
            order?: number;
          },
          index: number
        ) =>
          prisma.metric.update({
            where: { id: m.id },
            data: {
              ...(m.name && { name: m.name }),
              ...(m.description !== undefined && { description: m.description }),
              ...(m.type && { type: m.type }),
              ...(m.config !== undefined && {
                config: m.config ? JSON.stringify(m.config) : null,
              }),
              ...(m.isPrimary !== undefined && { isPrimary: m.isPrimary }),
              order: m.order ?? index,
            },
          })
      )
    );

    const updatedMetrics = await prisma.metric.findMany({
      where: { experimentId: id },
      orderBy: { order: "asc" },
    });

    return NextResponse.json(updatedMetrics);
  } catch (error) {
    console.error("Error updating metrics:", error);
    return NextResponse.json(
      { error: "Failed to update metrics" },
      { status: 500 }
    );
  }
}

// DELETE /api/experiments/[id]/metrics - Delete a metric (pass metricId in body)
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const user = await getOrCreateOwner();
    const body = await request.json();

    // Verify ownership
    const experiment = await prisma.experiment.findFirst({
      where: { id, userId: user.id },
    });

    if (!experiment) {
      return NextResponse.json(
        { error: "Experiment not found" },
        { status: 404 }
      );
    }

    const { metricId } = body;

    if (!metricId) {
      return NextResponse.json(
        { error: "Metric ID is required" },
        { status: 400 }
      );
    }

    await prisma.metric.delete({
      where: { id: metricId },
    });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error deleting metric:", error);
    return NextResponse.json(
      { error: "Failed to delete metric" },
      { status: 500 }
    );
  }
}
