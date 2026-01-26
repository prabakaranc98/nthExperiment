import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { getOrCreateOwner } from "@/lib/auth";

// GET /api/experiments - List all experiments
export async function GET(request: NextRequest) {
  try {
    const user = await getOrCreateOwner();
    const { searchParams } = new URL(request.url);

    const status = searchParams.get("status");
    const type = searchParams.get("type");

    const experiments = await prisma.experiment.findMany({
      where: {
        userId: user.id,
        ...(status && { status }),
        ...(type && { type }),
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

    return NextResponse.json(experiments);
  } catch (error) {
    console.error("Error fetching experiments:", error);
    return NextResponse.json(
      { error: "Failed to fetch experiments" },
      { status: 500 }
    );
  }
}

// POST /api/experiments - Create a new experiment
export async function POST(request: NextRequest) {
  try {
    const user = await getOrCreateOwner();
    const body = await request.json();

    const {
      title,
      hypothesis,
      type,
      intervention,
      successCriteria,
      frequency,
      customFrequency,
      startDate,
      endDate,
      tags,
      metrics,
    } = body;

    // Validate required fields
    if (!title || !hypothesis || !type) {
      return NextResponse.json(
        { error: "Title, hypothesis, and type are required" },
        { status: 400 }
      );
    }

    const experiment = await prisma.experiment.create({
      data: {
        userId: user.id,
        title,
        hypothesis,
        type,
        intervention,
        successCriteria,
        frequency: frequency || "daily",
        customFrequency: customFrequency
          ? JSON.stringify(customFrequency)
          : null,
        startDate: startDate ? new Date(startDate) : null,
        endDate: endDate ? new Date(endDate) : null,
        tags: tags ? JSON.stringify(tags) : null,
        status: "draft",
        metrics: metrics
          ? {
              create: metrics.map(
                (
                  m: {
                    name: string;
                    description?: string;
                    type: string;
                    config?: Record<string, unknown>;
                    isPrimary?: boolean;
                  },
                  index: number
                ) => ({
                  name: m.name,
                  description: m.description,
                  type: m.type,
                  config: m.config ? JSON.stringify(m.config) : null,
                  isPrimary: m.isPrimary || index === 0,
                  order: index,
                })
              ),
            }
          : undefined,
      },
      include: {
        metrics: {
          orderBy: { order: "asc" },
        },
      },
    });

    return NextResponse.json(experiment, { status: 201 });
  } catch (error) {
    console.error("Error creating experiment:", error);
    return NextResponse.json(
      { error: "Failed to create experiment" },
      { status: 500 }
    );
  }
}
