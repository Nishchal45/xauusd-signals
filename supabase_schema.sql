create extension if not exists pgcrypto;

create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text,
  role text not null default 'user',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.subscriptions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade,
  stripe_customer_id text,
  stripe_subscription_id text unique,
  status text not null default 'incomplete',
  price_id text,
  current_period_start timestamptz,
  current_period_end timestamptz,
  cancel_at_period_end boolean not null default false,
  raw jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.signals (
  id text primary key,
  alert_key text not null unique,
  source text,
  status text not null default 'OPEN',
  result text not null default 'OPEN',
  direction text not null check (direction in ('LONG', 'SHORT')),
  setup_type text,
  quality text,
  execution_timeframe text,
  generated_at timestamptz not null default now(),
  opened_bar text,
  valid_until timestamptz,
  invalidated_at timestamptz,
  invalidation_reason text,
  user_action text default 'PENDING' check (user_action in ('TAKEN', 'SKIPPED', 'PENDING')),
  user_action_at timestamptz,
  user_notes text,
  entry_low numeric(12, 3),
  entry_high numeric(12, 3),
  entry numeric(12, 3),
  stop_loss numeric(12, 3),
  take_profit_1 numeric(12, 3),
  take_profit_2 numeric(12, 3),
  risk numeric(12, 3),
  reward numeric(12, 3),
  risk_reward numeric(6, 2),
  position_size_1pct_10k_oz numeric(12, 4),
  weighted_score integer,
  confidence integer,
  h1_score integer,
  m15_score integer,
  m5_score integer,
  sweep_summary text,
  zone_low numeric(12, 3),
  zone_high numeric(12, 3),
  zone_score integer,
  reason text,
  closed_at timestamptz,
  closed_bar text,
  closed_price numeric(12, 3),
  observed_price numeric(12, 3),
  observed_high numeric(12, 3),
  observed_low numeric(12, 3),
  r_multiple numeric(8, 2),
  pnl_1pct_10k_usd numeric(12, 2),
  reasoning jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.user_signal_actions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  signal_id text not null references public.signals(id) on delete cascade,
  action text not null check (action in ('TAKEN', 'SKIPPED')),
  notes text,
  acted_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id, signal_id)
);

create table if not exists public.signal_outcomes (
  id uuid primary key default gen_random_uuid(),
  signal_id text not null references public.signals(id) on delete cascade,
  outcome text not null,
  outcome_price numeric(12, 3),
  outcome_at timestamptz not null default now(),
  pnl_r numeric(8, 2),
  raw jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.economic_events (
  id uuid primary key default gen_random_uuid(),
  ts timestamptz not null,
  country text,
  event text not null,
  impact text,
  actual numeric,
  forecast numeric,
  previous numeric,
  source text,
  raw jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.macro_features (
  ts timestamptz primary key default now(),
  source text not null default 'yfinance',
  status text,
  macro_score integer,
  macro_bias text,
  dxy numeric(12, 4),
  dxy_change numeric(12, 4),
  dxy_change_pct numeric(8, 4),
  us10y numeric(12, 4),
  us10y_change numeric(12, 4),
  us10y_change_pct numeric(8, 4),
  vix numeric(12, 4),
  vix_change numeric(12, 4),
  vix_change_pct numeric(8, 4),
  raw jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.ohlcv_xauusd (
  source text not null,
  instrument text not null,
  timeframe text not null,
  ts timestamptz not null,
  open numeric(12, 3) not null,
  high numeric(12, 3) not null,
  low numeric(12, 3) not null,
  close numeric(12, 3) not null,
  volume bigint,
  complete boolean not null default true,
  raw jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (source, instrument, timeframe, ts)
);

create table if not exists public.market_spreads (
  id uuid primary key default gen_random_uuid(),
  source text not null,
  instrument text not null,
  time timestamptz not null,
  bid numeric(12, 3) not null,
  ask numeric(12, 3) not null,
  mid numeric(12, 3) not null,
  spread_cents numeric(8, 2) not null,
  raw jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.stripe_events (
  id text primary key,
  type text not null,
  payload jsonb not null,
  created_at timestamptz not null default now()
);

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, email)
  values (new.id, new.email)
  on conflict (id) do update
    set email = excluded.email,
        updated_at = now();
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

create index if not exists idx_subscriptions_user_status
  on public.subscriptions(user_id, status, current_period_end desc);

create index if not exists idx_subscriptions_customer
  on public.subscriptions(stripe_customer_id);

create index if not exists idx_signals_generated
  on public.signals(generated_at desc);

create index if not exists idx_signals_setup
  on public.signals(execution_timeframe, direction, quality);

create index if not exists idx_signals_user_action
  on public.signals(user_action, generated_at desc);

create index if not exists idx_actions_user
  on public.user_signal_actions(user_id, acted_at desc);

create unique index if not exists idx_signal_outcomes_signal_id
  on public.signal_outcomes(signal_id);

create index if not exists idx_economic_events_ts
  on public.economic_events(ts);

create index if not exists idx_macro_features_ts
  on public.macro_features(ts desc);

create index if not exists idx_ohlcv_xauusd_lookup
  on public.ohlcv_xauusd(instrument, timeframe, ts desc);

create index if not exists idx_market_spreads_time
  on public.market_spreads(instrument, time desc);

alter table public.signals add column if not exists setup_type text;
alter table public.signals add column if not exists valid_until timestamptz;
alter table public.signals add column if not exists invalidated_at timestamptz;
alter table public.signals add column if not exists invalidation_reason text;
alter table public.signals add column if not exists user_action text default 'PENDING';
alter table public.signals add column if not exists user_action_at timestamptz;
alter table public.signals add column if not exists user_notes text;
alter table public.signals add column if not exists entry_low numeric(12, 3);
alter table public.signals add column if not exists entry_high numeric(12, 3);
alter table public.signals add column if not exists zone_low numeric(12, 3);
alter table public.signals add column if not exists zone_high numeric(12, 3);
alter table public.signals add column if not exists zone_score integer;
alter table public.macro_features add column if not exists source text default 'yfinance';
alter table public.macro_features add column if not exists status text;
alter table public.macro_features add column if not exists macro_score integer;
alter table public.macro_features add column if not exists macro_bias text;
alter table public.macro_features add column if not exists dxy numeric(12, 4);
alter table public.macro_features add column if not exists dxy_change numeric(12, 4);
alter table public.macro_features add column if not exists dxy_change_pct numeric(8, 4);
alter table public.macro_features add column if not exists us10y numeric(12, 4);
alter table public.macro_features add column if not exists us10y_change numeric(12, 4);
alter table public.macro_features add column if not exists us10y_change_pct numeric(8, 4);
alter table public.macro_features add column if not exists vix numeric(12, 4);
alter table public.macro_features add column if not exists vix_change numeric(12, 4);
alter table public.macro_features add column if not exists vix_change_pct numeric(8, 4);
alter table public.macro_features add column if not exists raw jsonb default '{}'::jsonb;
alter table public.macro_features add column if not exists created_at timestamptz default now();

alter table public.profiles enable row level security;
alter table public.subscriptions enable row level security;
alter table public.signals enable row level security;
alter table public.user_signal_actions enable row level security;
alter table public.signal_outcomes enable row level security;
alter table public.economic_events enable row level security;
alter table public.macro_features enable row level security;
alter table public.ohlcv_xauusd enable row level security;
alter table public.market_spreads enable row level security;
alter table public.stripe_events enable row level security;

drop policy if exists "profiles read own" on public.profiles;
drop policy if exists "profiles update own" on public.profiles;
drop policy if exists "subscriptions read own" on public.subscriptions;
drop policy if exists "active subscribers read signals" on public.signals;
drop policy if exists "active subscribers read outcomes" on public.signal_outcomes;
drop policy if exists "users read own signal actions" on public.user_signal_actions;
drop policy if exists "users insert own signal actions" on public.user_signal_actions;
drop policy if exists "users update own signal actions" on public.user_signal_actions;
drop policy if exists "active subscribers read economic events" on public.economic_events;
drop policy if exists "active subscribers read macro features" on public.macro_features;
drop policy if exists "active subscribers read ohlcv" on public.ohlcv_xauusd;
drop policy if exists "active subscribers read market spreads" on public.market_spreads;

create policy "profiles read own"
  on public.profiles for select
  using (id = auth.uid());

create policy "profiles update own"
  on public.profiles for update
  using (id = auth.uid())
  with check (id = auth.uid());

create policy "subscriptions read own"
  on public.subscriptions for select
  using (user_id = auth.uid());

create policy "active subscribers read signals"
  on public.signals for select
  using (
    exists (
      select 1
      from public.subscriptions s
      where s.user_id = auth.uid()
        and s.status in ('active', 'trialing')
        and (s.current_period_end is null or s.current_period_end > now())
    )
  );

create policy "active subscribers read outcomes"
  on public.signal_outcomes for select
  using (
    exists (
      select 1
      from public.subscriptions s
      where s.user_id = auth.uid()
        and s.status in ('active', 'trialing')
        and (s.current_period_end is null or s.current_period_end > now())
    )
  );

create policy "users read own signal actions"
  on public.user_signal_actions for select
  using (user_id = auth.uid());

create policy "users insert own signal actions"
  on public.user_signal_actions for insert
  with check (user_id = auth.uid());

create policy "users update own signal actions"
  on public.user_signal_actions for update
  using (user_id = auth.uid())
  with check (user_id = auth.uid());

create policy "active subscribers read economic events"
  on public.economic_events for select
  using (
    exists (
      select 1
      from public.subscriptions s
      where s.user_id = auth.uid()
        and s.status in ('active', 'trialing')
        and (s.current_period_end is null or s.current_period_end > now())
    )
  );

create policy "active subscribers read macro features"
  on public.macro_features for select
  using (
    exists (
      select 1
      from public.subscriptions s
      where s.user_id = auth.uid()
        and s.status in ('active', 'trialing')
        and (s.current_period_end is null or s.current_period_end > now())
    )
  );

create policy "active subscribers read ohlcv"
  on public.ohlcv_xauusd for select
  using (
    exists (
      select 1
      from public.subscriptions s
      where s.user_id = auth.uid()
        and s.status in ('active', 'trialing')
        and (s.current_period_end is null or s.current_period_end > now())
    )
  );

create policy "active subscribers read market spreads"
  on public.market_spreads for select
  using (
    exists (
      select 1
      from public.subscriptions s
      where s.user_id = auth.uid()
        and s.status in ('active', 'trialing')
        and (s.current_period_end is null or s.current_period_end > now())
    )
  );
