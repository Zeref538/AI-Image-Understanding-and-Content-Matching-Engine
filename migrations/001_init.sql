CREATE TABLE images (
    id         serial PRIMARY KEY,
    file       text NOT NULL UNIQUE,                 -- img-001.jpg: carries no label on purpose
    title      text,
    license    text NOT NULL,
    source_url text,
    sha256     text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- What the vision model said, after validation. One row per image (re-tagging replaces it).
CREATE TABLE image_metadata (
    image_id          integer PRIMARY KEY REFERENCES images (id) ON DELETE CASCADE,
    model             text NOT NULL,
    kind              text NOT NULL,
    subject           text NOT NULL,
    category          text NOT NULL,
    caption           text NOT NULL,
    confidence        real NOT NULL CHECK (confidence BETWEEN 0 AND 1),   -- the model's own claim
    kind_prob         real CHECK (kind_prob BETWEEN 0 AND 1),            -- measured from logprobs
    kind_alternatives jsonb NOT NULL DEFAULT '[]',
    flagged           boolean NOT NULL,
    flag_reason       text,
    raw               jsonb NOT NULL,
    tagged_at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX image_metadata_kind ON image_metadata (kind);
CREATE INDEX image_metadata_flagged ON image_metadata (image_id) WHERE flagged;

CREATE TABLE image_tags (
    image_id integer NOT NULL REFERENCES images (id) ON DELETE CASCADE,
    tag      text NOT NULL,
    PRIMARY KEY (image_id, tag)
);
CREATE INDEX image_tags_tag ON image_tags (tag);

CREATE TABLE posts (
    id         serial PRIMARY KEY,
    slug       text NOT NULL UNIQUE,
    title      text NOT NULL,
    body       text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE post_understanding (
    post_id  integer PRIMARY KEY REFERENCES posts (id) ON DELETE CASCADE,
    model    text NOT NULL,
    kind     text NOT NULL,
    subject  text NOT NULL,
    category text NOT NULL,
    summary  text NOT NULL,
    raw      jsonb NOT NULL,
    understood_at timestamptz NOT NULL DEFAULT now()
);

-- At 50 images an array and a Python dot product is plenty; pgvector is the upgrade path.
CREATE TABLE embeddings (
    owner_type text NOT NULL CHECK (owner_type IN ('image', 'post')),
    owner_id   integer NOT NULL,
    model      text NOT NULL,
    text       text NOT NULL,
    vector     float8[] NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (owner_type, owner_id, model)
);

CREATE TABLE suggestions (
    id         serial PRIMARY KEY,
    post_id    integer NOT NULL REFERENCES posts (id) ON DELETE CASCADE,
    image_id   integer NOT NULL REFERENCES images (id) ON DELETE CASCADE,
    rank       integer NOT NULL,
    similarity real NOT NULL,
    verdict    text NOT NULL CHECK (verdict IN ('accepted', 'rejected')),
    reasons    jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (post_id, image_id)
);
CREATE INDEX suggestions_post_rank ON suggestions (post_id, rank);

CREATE TABLE reviews (
    suggestion_id integer PRIMARY KEY REFERENCES suggestions (id) ON DELETE CASCADE,
    decision      text NOT NULL CHECK (decision IN ('approved', 'rejected')),
    note          text,
    reviewed_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE jobs (
    id              serial PRIMARY KEY,
    kind            text NOT NULL CHECK (kind IN ('tag_images', 'index_posts')),
    status          text NOT NULL DEFAULT 'queued'
                    CHECK (status IN ('queued', 'running', 'done', 'done_with_failures', 'stopped_budget')),
    idempotency_key text UNIQUE,
    force           boolean NOT NULL DEFAULT false,
    created_at      timestamptz NOT NULL DEFAULT now(),
    started_at      timestamptz,
    finished_at     timestamptz
);

CREATE TABLE job_items (
    id              serial PRIMARY KEY,
    job_id          integer NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    target_type     text NOT NULL CHECK (target_type IN ('image', 'post')),
    target_id       integer NOT NULL,
    status          text NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'done', 'flagged', 'failed', 'skipped')),
    attempts        integer NOT NULL DEFAULT 0,
    last_error      text,
    next_attempt_at timestamptz NOT NULL DEFAULT now(),
    finished_at     timestamptz,
    UNIQUE (job_id, target_type, target_id)
);
CREATE INDEX job_items_due ON job_items (next_attempt_at) WHERE status = 'pending';

-- The cost log: one row per model call, success or not.
CREATE TABLE ai_calls (
    id            serial PRIMARY KEY,
    job_id        integer REFERENCES jobs (id) ON DELETE SET NULL,
    purpose       text NOT NULL CHECK (purpose IN ('vision', 'post_understanding', 'embedding')),
    target_type   text NOT NULL,
    target_id     integer NOT NULL,
    model         text NOT NULL,
    input_tokens  integer NOT NULL DEFAULT 0,
    output_tokens integer NOT NULL DEFAULT 0,
    duration_ms   integer NOT NULL,
    cost_micros   bigint NOT NULL,              -- reference cloud price; actual local cost is 0
    status        text NOT NULL CHECK (status IN ('ok', 'invalid_output', 'error')),
    error         text,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ai_calls_job ON ai_calls (job_id);
CREATE INDEX ai_calls_target ON ai_calls (target_type, target_id);
CREATE INDEX ai_calls_day ON ai_calls (created_at);

CREATE TABLE alerts (
    id         serial PRIMARY KEY,
    kind       text NOT NULL,
    message    text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
