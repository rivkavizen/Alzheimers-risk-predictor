create table if not exists chat_sessions (
    id uuid primary key default gen_random_uuid(),
    assessment_id uuid not null references assessments (id) on delete cascade,
    created_at timestamptz not null default now()
);

create table if not exists chat_messages (
    id uuid primary key default gen_random_uuid(),
    session_id uuid not null references chat_sessions (id) on delete cascade,
    role text not null check (role in ('user', 'assistant')),
    content text not null,
    created_at timestamptz not null default now()
);

create index if not exists chat_messages_session_id_idx on chat_messages (session_id, created_at);
