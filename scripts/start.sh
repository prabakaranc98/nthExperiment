#!/bin/sh
set -e

echo "=== nthExperiment Startup ==="
echo "DATABASE_URL: $DATABASE_URL"

# Ensure data directory exists
mkdir -p /app/data

echo "Running database migrations..."

# Run prisma db push to create/sync tables
cd /app
npx prisma db push --skip-generate --accept-data-loss 2>&1 || {
    echo "Warning: prisma db push failed, trying again..."
    sleep 2
    npx prisma db push --skip-generate --accept-data-loss
}

echo "Database ready. Starting server..."

# Start the Next.js server
exec node server.js
