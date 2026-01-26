import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { getOrCreateOwner } from "@/lib/auth";

// PUT /api/entries/[id] - Update an entry
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const user = await getOrCreateOwner();
    const body = await request.json();

    // Find entry and verify ownership through experiment
    const existingEntry = await prisma.entry.findUnique({
      where: { id },
      include: { experiment: true },
    });

    if (!existingEntry || existingEntry.experiment.userId !== user.id) {
      return NextResponse.json({ error: "Entry not found" }, { status: 404 });
    }

    const { values, notes, mood } = body;

    const entry = await prisma.entry.update({
      where: { id },
      data: {
        ...(values && { values: JSON.stringify(values) }),
        ...(notes !== undefined && { notes }),
        ...(mood !== undefined && { mood }),
      },
    });

    return NextResponse.json(entry);
  } catch (error) {
    console.error("Error updating entry:", error);
    return NextResponse.json(
      { error: "Failed to update entry" },
      { status: 500 }
    );
  }
}

// DELETE /api/entries/[id] - Delete an entry
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const user = await getOrCreateOwner();

    // Find entry and verify ownership through experiment
    const existingEntry = await prisma.entry.findUnique({
      where: { id },
      include: { experiment: true },
    });

    if (!existingEntry || existingEntry.experiment.userId !== user.id) {
      return NextResponse.json({ error: "Entry not found" }, { status: 404 });
    }

    await prisma.entry.delete({
      where: { id },
    });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error deleting entry:", error);
    return NextResponse.json(
      { error: "Failed to delete entry" },
      { status: 500 }
    );
  }
}
