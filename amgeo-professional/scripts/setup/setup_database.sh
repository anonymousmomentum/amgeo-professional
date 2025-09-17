#!/bin/bash
# Setup development database

set -e

echo "Setting up development database..."

# Start PostgreSQL service if not running
sudo systemctl start postgresql || true

# Wait for PostgreSQL to be ready
until pg_isready -h localhost -p 5432; do
  echo "Waiting for PostgreSQL..."
  sleep 2
done

# Create database and user
sudo -u postgres psql << EOF
DO \$\$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'amdigits') THEN
      CREATE DATABASE amdigits;
   END IF;
END\$\$;
EOF

sudo -u postgres psql -d amdigits << EOF
DO \$\$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'anonmomentum') THEN
      CREATE ROLE anonmomentum WITH LOGIN PASSWORD 'mapple';
   END IF;
END\$\$;
GRANT ALL PRIVILEGES ON DATABASE amdigits TO anonmomentum;
ALTER ROLE anonmomentum CREATEDB;
EOF
