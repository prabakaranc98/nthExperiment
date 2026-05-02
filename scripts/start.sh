#!/bin/sh
set -e

echo "=== nthExperiment Startup ==="

# Check if using local SQLite or remote Turso
if echo "$DATABASE_URL" | grep -q "^file:"; then
    echo "Using local SQLite database"
    # Ensure data directory exists for local SQLite
    mkdir -p /app/data

    echo "Running database migrations..."
    cd /app
    npx prisma db push --skip-generate --accept-data-loss 2>&1 || {
        echo "Warning: prisma db push failed, trying again..."
        sleep 2
        npx prisma db push --skip-generate --accept-data-loss
    }
else
    echo "Using remote database: $DATABASE_URL"
    echo "Running database migrations..."
    cd /app
    npx prisma db push --skip-generate --accept-data-loss 2>&1 || {
        echo "Warning: prisma db push failed, trying again..."
        sleep 2
        npx prisma db push --skip-generate --accept-data-loss
    }
fi

echo "Database ready. Starting server..."

# Start the Next.js server
exec node server.js
