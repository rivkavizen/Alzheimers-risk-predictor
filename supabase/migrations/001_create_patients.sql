create extension if not exists "pgcrypto";

create table if not exists patients (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    date_of_birth date,
    notes text,
    created_at timestamptz not null default now()
);

create index if not exists patients_created_at_idx on patients (created_at desc);
