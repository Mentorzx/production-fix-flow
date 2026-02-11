-- PFF Database Initialization Script
-- Runs automatically on first container startup

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS pff;

-- Set search path
SET search_path TO pff, public;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA pff TO pff_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA pff TO pff_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA pff TO pff_user;

-- Create tables (fallback if alembic is not used)
CREATE TABLE IF NOT EXISTS public.kg_splits (
    id BIGSERIAL PRIMARY KEY,
    s TEXT NOT NULL,
    p TEXT NOT NULL,
    o TEXT NOT NULL,
    split_name TEXT NOT NULL,
    split_type TEXT NOT NULL,
    sample_id TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE '✅ PFF database initialized successfully';
    RAISE NOTICE '✅ Extensions: vector, pg_trgm, pg_stat_statements';
    RAISE NOTICE '✅ Schema: pff';
    RAISE NOTICE '🔄 Run alembic migrations next: alembic upgrade head';
END $$;
