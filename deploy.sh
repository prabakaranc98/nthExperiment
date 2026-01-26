#!/bin/bash

# nthExperiment - Cloud Run Deployment Script
# Usage: ./deploy.sh <project-id> <your-email>

set -e

PROJECT_ID=${1:-"your-project-id"}
OWNER_EMAIL=${2:-"owner@example.com"}
REGION="us-central1"
SERVICE_NAME="nthexperiment"

echo "Deploying nthExperiment to Cloud Run..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Owner: $OWNER_EMAIL"

# Set project
gcloud config set project $PROJECT_ID

# Build image
echo "Building container image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 1 \
  --max-instances 1 \
  --allow-unauthenticated \
  --set-env-vars "OWNER_EMAIL=$OWNER_EMAIL,NODE_ENV=production"

# Get URL
URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')
echo ""
echo "Deployed successfully!"
echo "URL: $URL"
