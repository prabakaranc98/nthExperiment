import { prisma } from "./db";
import { cookies } from "next/headers";

const OWNER_EMAIL = process.env.OWNER_EMAIL || "owner@example.com";
const AUTH_TOKEN_NAME = "nthexp_auth";

// For MVP, we use a simple token-based auth
// The owner email is set via environment variable
export async function getOrCreateOwner() {
  let user = await prisma.user.findUnique({
    where: { email: OWNER_EMAIL },
  });

  if (!user) {
    user = await prisma.user.create({
      data: {
        email: OWNER_EMAIL,
        name: "Owner",
      },
    });
  }

  return user;
}

export async function getCurrentUser() {
  const cookieStore = await cookies();
  const authToken = cookieStore.get(AUTH_TOKEN_NAME);

  if (!authToken?.value) {
    return null;
  }

  // For MVP, the token is just the user ID
  const user = await prisma.user.findUnique({
    where: { id: authToken.value },
  });

  return user;
}

export async function isAuthenticated() {
  const user = await getCurrentUser();
  return user !== null;
}

export function generateAuthToken(userId: string) {
  // For MVP, just use the user ID as token
  // In production, use proper JWT or session tokens
  return userId;
}

export { AUTH_TOKEN_NAME };
