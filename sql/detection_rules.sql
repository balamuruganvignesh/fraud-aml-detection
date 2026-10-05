-- Rule-based AML detectors. Each block starts with "-- name: <rule>" and must return
-- columns: account_id, score_hint, details (JSON text), txn_ids (comma-separated).
-- Named parameters (:param) are bound from DetectionConfig.

-- name: structuring
-- One account repeatedly sending amounts just under the reporting threshold within 7 days.
WITH banded AS (
    SELECT t.*,
           COUNT(*) OVER w AS n_in_window
    FROM transactions t
    WHERE amount >= :structuring_low AND amount < :structuring_high
    WINDOW w AS (PARTITION BY sender ORDER BY julianday(timestamp)
                 RANGE BETWEEN 7.0 PRECEDING AND CURRENT ROW)
)
SELECT sender                                      AS account_id,
       MAX(n_in_window)                            AS score_hint,
       json_object('max_txns_in_7d', MAX(n_in_window),
                   'total_amount', ROUND(SUM(amount), 2)) AS details,
       group_concat(txn_id)                        AS txn_ids
FROM banded
GROUP BY sender
HAVING MAX(n_in_window) >= :structuring_min_count;

-- name: structuring_receiver
-- Many distinct senders depositing sub-threshold amounts into the same account within 7 days.
WITH banded AS (
    SELECT * FROM transactions
    WHERE amount >= :structuring_low AND amount < :structuring_high
),
windows AS (   -- one 7-day window anchored on each banded deposit
    SELECT a.receiver, a.txn_id AS anchor,
           COUNT(DISTINCT b.sender) AS n_senders,
           SUM(b.amount)            AS total
    FROM banded a
    JOIN banded b
      ON b.receiver = a.receiver
     AND julianday(b.timestamp) BETWEEN julianday(a.timestamp) AND julianday(a.timestamp) + 7
    GROUP BY a.txn_id
)
SELECT w.receiver                                  AS account_id,
       MAX(w.n_senders)                            AS score_hint,
       json_object('max_distinct_senders_7d', MAX(w.n_senders),
                   'max_amount_7d', ROUND(MAX(w.total), 2)) AS details,
       (SELECT group_concat(x.txn_id) FROM banded x WHERE x.receiver = w.receiver) AS txn_ids
FROM windows w
GROUP BY w.receiver
HAVING MAX(w.n_senders) >= :structuring_min_count;

-- name: rapid_pass_through
-- Money leaves an account shortly after it arrives, in a similar amount (mule behaviour).
WITH pairs AS (
    SELECT i.receiver AS acct, i.txn_id AS in_id, i.amount AS in_amt,
           o.txn_id   AS out_id, o.amount AS out_amt
    FROM transactions i
    JOIN transactions o
      ON o.sender = i.receiver
     AND o.receiver <> i.sender
     AND julianday(o.timestamp) >  julianday(i.timestamp)
     AND julianday(o.timestamp) <= julianday(i.timestamp) + :pass_through_hours / 24.0
     AND o.amount BETWEEN i.amount * :pass_through_min_ratio AND i.amount
    WHERE i.amount >= :pass_through_min_amount
),
ins  AS (SELECT DISTINCT acct, in_id,  in_amt  FROM pairs),
outs AS (SELECT DISTINCT acct, out_id, out_amt FROM pairs)
SELECT ins.acct                                    AS account_id,
       COUNT(*)                                    AS score_hint,
       json_object('matched_inflows', COUNT(*),
                   'amount_in',  ROUND(SUM(ins.in_amt), 2),
                   'amount_out', (SELECT ROUND(SUM(out_amt), 2) FROM outs WHERE outs.acct = ins.acct)) AS details,
       group_concat(ins.in_id) || ',' ||
       (SELECT group_concat(out_id) FROM outs WHERE outs.acct = ins.acct) AS txn_ids
FROM ins
GROUP BY ins.acct
HAVING COUNT(*) >= :pass_through_min_pairs;

-- name: high_velocity
-- Burst of outgoing transfers in a short window.
WITH w AS (
    SELECT t.*,
           COUNT(*) OVER (PARTITION BY sender ORDER BY julianday(timestamp)
                          RANGE BETWEEN :velocity_window_days PRECEDING AND CURRENT ROW) AS n_window
    FROM transactions t
    WHERE channel <> 'card'
)
SELECT sender                                      AS account_id,
       MAX(n_window)                               AS score_hint,
       json_object('max_txns_in_window', MAX(n_window),
                   'window_hours', :velocity_window_hours) AS details,
       group_concat(txn_id)                        AS txn_ids
FROM w
GROUP BY sender
HAVING MAX(n_window) >= :velocity_min_txns
   AND sender NOT IN (SELECT account_id FROM accounts WHERE account_type = 'business'
                      AND account_id IN (SELECT sender FROM transactions WHERE channel = 'ach'
                                         GROUP BY sender HAVING COUNT(DISTINCT receiver) >= 5));

-- name: high_risk_jurisdiction
-- Large transfers to or from accounts domiciled in high-risk jurisdictions.
SELECT CASE WHEN rs.country IN (SELECT value FROM json_each(:high_risk_countries))
            THEN t.sender ELSE t.receiver END       AS account_id,
       COUNT(*)                                     AS score_hint,
       json_object('n_transfers', COUNT(*),
                   'total_amount', ROUND(SUM(t.amount), 2),
                   'counterparty_countries', group_concat(DISTINCT
                        CASE WHEN rs.country IN (SELECT value FROM json_each(:high_risk_countries))
                             THEN rs.country ELSE ss.country END)) AS details,
       group_concat(t.txn_id)                       AS txn_ids
FROM transactions t
JOIN accounts ss ON ss.account_id = t.sender
JOIN accounts rs ON rs.account_id = t.receiver
WHERE t.amount >= :high_risk_min_amount
  AND (rs.country IN (SELECT value FROM json_each(:high_risk_countries))
       OR ss.country IN (SELECT value FROM json_each(:high_risk_countries)))
GROUP BY 1;
