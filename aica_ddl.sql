-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS aica_schema;

-- Create extensions in public schema
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" SCHEMA public;

-- Create users table in aica_schema
CREATE TABLE IF NOT EXISTS aica_schema.users (
    id uuid PRIMARY KEY DEFAULT public.uuid_generate_v4(),
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create api_keys table in aica_schema
CREATE TABLE IF NOT EXISTS aica_schema.api_keys (
    id uuid PRIMARY KEY DEFAULT public.uuid_generate_v4(),
    customer_id TEXT NOT NULL,
    api_key TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Enable Row Level Security
ALTER TABLE aica_schema.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE aica_schema.api_keys ENABLE ROW LEVEL SECURITY;

-- Create RLS Policies
-- Drop existing policies
DROP POLICY IF EXISTS "Users can read their own data" ON aica_schema.users;
DROP POLICY IF EXISTS "API keys are readable by their owners" ON aica_schema.api_keys;

-- Create new RLS policies using session_user
CREATE POLICY "Users can read their own data"
    ON aica_schema.users
    FOR SELECT
    USING (id::text = current_setting('app.current_user_id', true));

CREATE POLICY "API keys are readable by their owners"
    ON aica_schema.api_keys
    FOR SELECT
    USING (customer_id IN (
        SELECT customer_id 
        FROM aica_schema.users 
        WHERE id::text = current_setting('app.current_user_id', true)
    ));