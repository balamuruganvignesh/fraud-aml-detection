-- Core banking tables -------------------------------------------------------
CREATE TABLE IF NOT EXISTS accounts (
    account_id    TEXT PRIMARY KEY,
    account_type  TEXT NOT NULL,              -- individual | business | merchant
    country       TEXT NOT NULL,              -- ISO-3166 alpha-2
    open_date     TEXT NOT NULL,
    kyc_risk      TEXT NOT NULL,              -- low | medium | high
    is_laundering INTEGER NOT NULL DEFAULT 0  -- ground-truth label (synthetic data only)
);

CREATE TABLE IF NOT EXISTS transactions (
    txn_id        TEXT PRIMARY KEY,
    timestamp     TEXT NOT NULL,              -- 'YYYY-MM-DD HH:MM:SS'
    sender        TEXT NOT NULL REFERENCES accounts(account_id),
    receiver      TEXT NOT NULL REFERENCES accounts(account_id),
    amount        REAL NOT NULL,
    channel       TEXT NOT NULL,              -- wire | ach | card | p2p | cash_deposit
    is_laundering INTEGER NOT NULL DEFAULT 0, -- ground-truth label (synthetic data only)
    typology      TEXT,
    ring_id       TEXT
);

CREATE INDEX IF NOT EXISTS idx_txn_sender   ON transactions(sender, timestamp);
CREATE INDEX IF NOT EXISTS idx_txn_receiver ON transactions(receiver, timestamp);
CREATE INDEX IF NOT EXISTS idx_txn_amount   ON transactions(amount);

-- Detection output -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS rule_hits (
    rule        TEXT NOT NULL,
    account_id  TEXT NOT NULL,
    score_hint  REAL,
    details     TEXT,                         -- JSON
    txn_ids     TEXT                          -- comma-separated evidence
);

CREATE TABLE IF NOT EXISTS cases (
    case_id         TEXT PRIMARY KEY,
    risk_score      REAL NOT NULL,
    typologies      TEXT NOT NULL,            -- comma-separated
    n_accounts      INTEGER NOT NULL,
    n_transactions  INTEGER NOT NULL,
    total_amount    REAL NOT NULL,
    is_alert        INTEGER NOT NULL,
    llm_source      TEXT,                     -- gemini model name or 'rules-fallback'
    llm_flag        INTEGER,
    llm_risk_level  TEXT,
    llm_confidence  REAL,
    llm_summary     TEXT,
    llm_json        TEXT,
    disposition     TEXT,                     -- escalate | review | close (analyst work queue)
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS case_accounts (
    case_id     TEXT NOT NULL REFERENCES cases(case_id),
    account_id  TEXT NOT NULL REFERENCES accounts(account_id),
    role        TEXT,
    PRIMARY KEY (case_id, account_id)
);

CREATE TABLE IF NOT EXISTS case_transactions (
    case_id  TEXT NOT NULL REFERENCES cases(case_id),
    txn_id   TEXT NOT NULL REFERENCES transactions(txn_id),
    PRIMARY KEY (case_id, txn_id)
);
