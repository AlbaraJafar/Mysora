-- Mysora Supabase Schema
-- Run this in the Supabase SQL Editor (Dashboard → SQL Editor → New query).
-- Do NOT run programmatically — execute manually once per project.

-- ============================================================
-- TABLES
-- ============================================================

create table public.users (
  id uuid references auth.users(id) primary key,
  role text not null check (role in ('student', 'teacher', 'admin')),
  display_name text not null,
  created_at timestamptz default now()
);



create table public.classes (
  id uuid default gen_random_uuid() primary key,
  teacher_id uuid references public.users(id) not null,
  name text not null,
  join_code text unique not null,
  created_at timestamptz default now()
);

create table public.class_students (
  class_id uuid references public.classes(id) not null,
  student_id uuid references public.users(id) not null,
  joined_at timestamptz default now(),
  primary key (class_id, student_id)
);

create table public.practice_sessions (
  id uuid default gen_random_uuid() primary key,
  student_id uuid references public.users(id) not null,
  letter text not null,
  confidence_tier text not null,
  correct boolean not null,
  created_at timestamptz default now()
);

-- ============================================================
-- INDEXES
-- ============================================================

create index idx_practice_sessions_student on public.practice_sessions(student_id);
create index idx_practice_sessions_created on public.practice_sessions(created_at desc);
create index idx_class_students_class on public.class_students(class_id);
create index idx_classes_teacher on public.classes(teacher_id);

-- ============================================================
-- ROW LEVEL SECURITY
-- ============================================================

alter table public.users enable row level security;
alter table public.practice_sessions enable row level security;
alter table public.classes enable row level security;
alter table public.class_students enable row level security;

-- Users can read their own profile
create policy "Users read own profile"
  on public.users for select
  using (auth.uid() = id);

-- Users can insert their own profile (on sign-up trigger or client)
create policy "Users insert own profile"
  on public.users for insert
  with check (auth.uid() = id);

-- Students can read their own sessions
create policy "Students read own sessions"
  on public.practice_sessions for select
  using (auth.uid() = student_id);

-- Students can insert their own sessions (client-side, optional)
create policy "Students insert own sessions"
  on public.practice_sessions for insert
  with check (auth.uid() = student_id);

-- Teachers can read their own classes
create policy "Teachers read own classes"
  on public.classes for select
  using (auth.uid() = teacher_id);

-- Students can read classes they belong to
create policy "Students read joined classes"
  on public.class_students for select
  using (auth.uid() = student_id);

-- Service role bypasses RLS entirely (used by the Mysora backend
-- with SUPABASE_SERVICE_KEY for cross-user queries by teachers).

-- ============================================================
-- TRIGGER: auto-create user profile on sign-up
-- ============================================================
-- Optional: if you want the profile row created automatically
-- when a user signs up via Supabase Auth, add this trigger.
-- Requires passing display_name and role in raw_user_meta_data
-- from the client during signUp().

create or replace function public.handle_new_user()
returns trigger language plpgsql security definer as $$
begin
  insert into public.users (id, role, display_name)
  values (
    new.id,
    coalesce(new.raw_user_meta_data->>'role', 'student'),
    coalesce(new.raw_user_meta_data->>'display_name', split_part(new.email, '@', 1))
  );
  return new;
end;
$$;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();
