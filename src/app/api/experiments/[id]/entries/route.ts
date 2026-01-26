import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { getOrCreateOwner } from "@/lib/auth";

// GET /api/experiments/[id]/entries - List entries for an experiment
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const user = await getOrCreateOwner();

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

    const entries = await prisma.entry.findMany({
      where: { experimentId: id },
      orderBy: { date: "desc" },
    });

    return NextResponse.json(entries);
  } catch (error) {
    console.error("Error fetching entries:", error);
    return NextResponse.json(
      { error: "Failed to fetch entries" },
      { status: 500 }
    );
  }
}

// POST /api/experiments/[id]/entries - Create a new entry
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
    });

    if (!experiment) {
      return NextResponse.json(
        { error: "Experiment not found" },
        { status: 404 }
      );
    }

    const { date, values, notes, mood } = body;

    if (!date || !values) {
      return NextResponse.json(
        { error: "Date and values are required" },
        { status: 400 }
      );
    }

    // Normalize the date to start of day in UTC
    const entryDate = new Date(date);
    entryDate.setUTCHours(0, 0, 0, 0);

    // Check if entry already exists for this date
    const existingEntry = await prisma.entry.findFirst({
      where: {
        experimentId: id,
        date: entryDate,
      },
    });

    let entry;
    if (existingEntry) {
      // Update existing entry
      entry = await prisma.entry.update({
        where: { id: existingEntry.id },
        data: {
          values: JSON.stringify(values),
          notes,
          mood,
        },
      });
    } else {
      // Create new entry
      entry = await prisma.entry.create({
        data: {
          experimentId: id,
          date: entryDate,
          values: JSON.stringify(values),
          notes,
          mood,
        },
      });
    }

    return NextResponse.json(entry, { status: existingEntry ? 200 : 201 });
  } catch (error) {
    console.error("Error creating entry:", error);
    return NextResponse.json(
      { error: "Failed to create entry" },
      { status: 500 }
    );
  }
}
