# Supabase setup

**Project dashboard:** [puhtwpurgkqylnanortj](https://supabase.com/dashboard/project/puhtwpurgkqylnanortj)

**Project URL:** `https://puhtwpurgkqylnanortj.supabase.co`

## 1. Run migrations (SQL Editor)

Run in order:

1. `migrations/001_create_patients.sql`
2. `migrations/002_create_assessments.sql`
3. `migrations/003_create_chat_sessions.sql`

## 2. API key for local / Railway

1. Open [Settings → API](https://supabase.com/dashboard/project/puhtwpurgkqylnanortj/settings/api)
2. Copy the **service_role** key (secret)
3. Paste into project root `.env` as `SUPABASE_KEY=...`
4. Restart Flask (`python app.py`)

## 3. Verify connection

```powershell
cd backend
python -c "from db import get_supabase; print(get_supabase().table('patients').select('id').limit(1).execute())"
```

If migrations ran and the key is set, this returns without error (empty list is OK).
