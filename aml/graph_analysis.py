"""NetworkX transaction graph and graph-based laundering detectors.

Accounts are nodes; every transaction is an edge in a ``MultiDiGraph`` and is also rolled
up into a weighted ``DiGraph`` (total amount, count, max single amount per pair). Detectors
combine graph structure with time ordering, because laundering is a *temporal* pattern:
a cycle only matters if the money actually travels around it in sequence.
"""

from __future__ import annotations

import bisect
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

from .config import DetectionConfig
from .models import Finding

HOUR = 3600.0
DAY = 86_400.0


@dataclass(frozen=True, order=True)
class Txn:
    ts: float
    amount: float
    sender: str
    receiver: str
    txn_id: str
    channel: str = field(default="unknown", compare=False)


class TransactionGraph:
    def __init__(self, accounts: pd.DataFrame, transactions: pd.DataFrame):
        self.multi = nx.MultiDiGraph()
        self.agg = nx.DiGraph()
        for row in accounts.to_dict("records"):
            attrs = {k: v for k, v in row.items() if k != "account_id"}
            self.multi.add_node(row["account_id"], **attrs)
            self.agg.add_node(row["account_id"], **attrs)

        # Resolution-independent epoch seconds (pandas may parse to s, ms, us or ns).
        epoch = (pd.to_datetime(transactions["timestamp"]) - pd.Timestamp(0)) // pd.Timedelta(seconds=1)
        self.txns: dict[str, Txn] = {}
        self.out_txns: dict[str, list[Txn]] = defaultdict(list)
        self.in_txns: dict[str, list[Txn]] = defaultdict(list)
        self.edge_txns: dict[tuple[str, str], list[Txn]] = defaultdict(list)

        for ts, row in zip(epoch, transactions.itertuples(index=False)):
            t = Txn(float(ts), float(row.amount), row.sender, row.receiver, row.txn_id,
                    getattr(row, "channel", "unknown") or "unknown")
            self.txns[t.txn_id] = t
            self.out_txns[t.sender].append(t)
            self.in_txns[t.receiver].append(t)
            self.edge_txns[(t.sender, t.receiver)].append(t)
            self.multi.add_edge(t.sender, t.receiver, key=t.txn_id, amount=t.amount, ts=t.ts)
            if self.agg.has_edge(t.sender, t.receiver):
                d = self.agg[t.sender][t.receiver]
                d["weight"] += t.amount
                d["count"] += 1
                d["max_amount"] = max(d["max_amount"], t.amount)
            else:
                self.agg.add_edge(t.sender, t.receiver, weight=t.amount, count=1, max_amount=t.amount)

        # Freeze to plain dicts so lookups of unknown keys never mutate the index mid-iteration.
        self.out_txns = {k: sorted(v) for k, v in self.out_txns.items()}
        self.in_txns = {k: sorted(v) for k, v in self.in_txns.items()}
        self.edge_txns = {k: sorted(v) for k, v in self.edge_txns.items()}
        self._out_ts = {a: [t.ts for t in lst] for a, lst in self.out_txns.items()}
        self._pagerank: dict[str, float] | None = None
        self.observation_start = pd.Timestamp(min(epoch), unit="s") if len(epoch) else None
        self._pagerank_pct: dict[str, float] = {}
        self._communities: dict[str, int] | None = None
        self._community_sizes: Counter | None = None

    # ------------------------------------------------------------------ helpers
    def outgoing_between(self, account: str, start: float, end: float) -> list[Txn]:
        """Outgoing transactions of ``account`` with start < ts <= end."""
        ts = self._out_ts.get(account, [])
        lo = bisect.bisect_right(ts, start)
        hi = bisect.bisect_right(ts, end)
        return self.out_txns.get(account, [])[lo:hi]

    def degree(self, account: str) -> int:
        """Number of distinct counterparties (in or out)."""
        if account not in self.agg:
            return 0
        return len(set(self.agg.predecessors(account)) | set(self.agg.successors(account)))

    @property
    def pagerank(self) -> dict[str, float]:
        if self._pagerank is None:
            self._pagerank = nx.pagerank(self.agg, weight="weight")
            ranked = sorted(self._pagerank, key=self._pagerank.get)
            self._pagerank_pct = {a: (i + 1) / len(ranked) * 100 for i, a in enumerate(ranked)}
        return self._pagerank

    def communities(self, seed: int = 42) -> dict[str, int]:
        """Louvain communities on the undirected, amount-weighted graph."""
        if self._communities is None:
            und = nx.Graph()
            und.add_nodes_from(self.agg.nodes)
            for u, v, d in self.agg.edges(data=True):
                w = und[u][v]["weight"] + d["weight"] if und.has_edge(u, v) else d["weight"]
                und.add_edge(u, v, weight=w)
            parts = nx.community.louvain_communities(und, weight="weight", seed=seed)
            self._communities = {n: i for i, part in enumerate(parts) for n in part}
            self._community_sizes = Counter(self._communities.values())
        return self._communities

    def account_profile(self, account: str) -> dict:
        node = self.agg.nodes[account]
        ins, outs = self.in_txns.get(account, []), self.out_txns.get(account, [])
        self.pagerank  # ensure percentiles are computed
        community = self.communities().get(account)
        return {
            "account_id": account,
            "account_type": node.get("account_type"),
            "country": node.get("country"),
            "kyc_risk": node.get("kyc_risk"),
            "open_date": node.get("open_date"),
            "n_in": len(ins),
            "n_out": len(outs),
            "total_in": round(sum(t.amount for t in ins), 2),
            "total_out": round(sum(t.amount for t in outs), 2),
            "unique_senders": len({t.sender for t in ins}),
            "unique_receivers": len({t.receiver for t in outs}),
            "pagerank_percentile": round(self._pagerank_pct.get(account, 0.0), 1),
            "community_id": community,
            "community_size": self._community_sizes[community] if community is not None else None,
            **self._behaviour(account, node, ins, outs),
        }

    def _behaviour(self, account: str, node: dict, ins: list[Txn], outs: list[Txn]) -> dict:
        """Context that helps tell ordinary customers from purpose-built mule or shell accounts."""
        age = None
        if node.get("open_date") and self.observation_start is not None:
            age = (self.observation_start.normalize() - pd.Timestamp(node["open_date"])).days
        nodes = self.agg.nodes
        salary_in = [t for t in ins if t.channel == "ach" and nodes[t.sender].get("account_type") == "business"]
        payroll_out = {t.receiver for t in outs if t.channel == "ach"}
        return {
            "account_age_days": age,
            "receives_salary": bool(salary_in),
            "runs_payroll_to_n_people": len(payroll_out) if len(payroll_out) >= 5 else 0,
            "everyday_card_purchases": sum(t.channel == "card" for t in outs),
            "cash_deposits_received": sum(t.channel == "cash_deposit" for t in ins),
        }

    def subgraph_edges(self, accounts: list[str], txn_ids: list[str]) -> list[dict]:
        """Aggregated edges among case accounts, limited to the case's evidence transactions."""
        wanted = set(txn_ids)
        edges: dict[tuple[str, str], dict] = {}
        for tid in wanted:
            t = self.txns.get(tid)
            if t is None:
                continue
            e = edges.setdefault((t.sender, t.receiver), {"source": t.sender, "target": t.receiver,
                                                          "amount": 0.0, "count": 0})
            e["amount"] += t.amount
            e["count"] += 1
        return list(edges.values())


# ======================================================================= detectors
def detect_cycles(g: TransactionGraph, cfg: DetectionConfig) -> list[Finding]:
    """Round-tripping: funds that travel A -> B -> ... -> A in time order, mostly intact."""
    strong = nx.DiGraph()
    strong.add_edges_from((u, v) for u, v, d in g.agg.edges(data=True)
                          if d["max_amount"] >= cfg.cycle_min_amount and u != v)
    findings = []
    for cycle in nx.simple_cycles(strong, length_bound=cfg.cycle_max_length):
        if len(cycle) < 2:
            continue
        loops = _temporal_cycle_matches(g, cycle, cfg)
        if not loops:
            continue
        txn_ids = sorted({t.txn_id for loop in loops for t in loop})
        first = loops[0]
        retention = first[-1].amount / first[0].amount
        hours = (first[-1].ts - first[0].ts) / HOUR
        findings.append(Finding(
            detector="cycle", source="graph", accounts=list(cycle), txn_ids=txn_ids,
            strength=min(1.0, 0.6 + 0.1 * len(cycle) + 0.1 * (len(loops) - 1)),
            details={"cycle_length": len(cycle), "loops_completed": len(loops),
                     "start_amount": round(first[0].amount, 2), "end_amount": round(first[-1].amount, 2),
                     "retention": round(retention, 3), "loop_hours": round(hours, 1),
                     "path": " -> ".join([*cycle, cycle[0]])},
            roles={a: "cycle_member" for a in cycle},
        ))
    return findings


def _temporal_cycle_matches(g: TransactionGraph, cycle: list[str], cfg: DetectionConfig) -> list[list[Txn]]:
    """All non-overlapping time-ordered traversals of ``cycle`` (any rotation)."""
    matches, used = [], set()
    n = len(cycle)
    for r in range(n):
        nodes = cycle[r:] + cycle[:r]
        hops = [(nodes[i], nodes[(i + 1) % n]) for i in range(n)]
        for first in g.edge_txns.get(hops[0], []):
            if first.amount < cfg.cycle_min_amount or first.txn_id in used:
                continue
            path = [first]
            for hop in hops[1:]:
                prev = path[-1]
                nxt = next((t for t in g.edge_txns.get(hop, [])
                            if t.ts > prev.ts and t.txn_id not in used
                            and t.ts - first.ts <= cfg.cycle_max_days * DAY
                            and cfg.cycle_min_retention * prev.amount <= t.amount <= 1.05 * prev.amount), None)
                if nxt is None:
                    break
                path.append(nxt)
            if len(path) == n:
                matches.append(path)
                used.update(t.txn_id for t in path)
    return matches


def detect_layering_chains(g: TransactionGraph, cfg: DetectionConfig) -> list[Finding]:
    """Rapid multi-hop pass-through: A -> B -> C -> D, each hop soon after the last, amount ~kept."""
    window = cfg.chain_max_hours_per_hop * HOUR
    chains: list[list[Txn]] = []

    def extend(path: list[Txn], visited: set[str]) -> None:
        last = path[-1]
        nexts = [t for t in g.outgoing_between(last.receiver, last.ts, last.ts + window)
                 if t.receiver not in visited
                 and cfg.chain_min_retention * last.amount <= t.amount <= last.amount]
        if not nexts:
            if len(path) >= cfg.chain_min_hops:
                chains.append(list(path))
            return
        for t in nexts[:3]:  # bound branching on busy accounts
            path.append(t)
            visited.add(t.receiver)
            extend(path, visited)
            visited.discard(t.receiver)
            path.pop()

    for start in sorted(g.txns.values()):
        if start.amount >= cfg.chain_min_amount:
            extend([start], {start.sender, start.receiver})

    # keep only maximal chains (drop any chain whose transactions are inside a longer one)
    chains.sort(key=len, reverse=True)
    kept: list[tuple[frozenset, list[Txn]]] = []
    for chain in chains:
        ids = frozenset(t.txn_id for t in chain)
        if not any(ids <= k for k, _ in kept):
            kept.append((ids, chain))

    findings = []
    for ids, chain in kept:
        path = [chain[0].sender] + [t.receiver for t in chain]
        hours = (chain[-1].ts - chain[0].ts) / HOUR
        roles = {a: "intermediary" for a in path[1:-1]}
        roles[path[0]], roles[path[-1]] = "origin", "exit"
        findings.append(Finding(
            detector="layering_chain", source="graph", accounts=path, txn_ids=sorted(ids),
            strength=min(1.0, 0.5 + 0.1 * len(chain) + (0.1 if hours < 72 else 0)),
            details={"hops": len(chain), "path": " -> ".join(path), "total_hours": round(hours, 1),
                     "start_amount": round(chain[0].amount, 2), "end_amount": round(chain[-1].amount, 2)},
            roles=roles,
        ))
    return findings


def _best_window(txns: list[Txn], counterparty, span: float, min_parties: int, min_total: float):
    """Sliding window over time-sorted txns; return the max-total window meeting the thresholds."""
    best, window, parties, total = None, deque(), Counter(), 0.0
    for t in txns:
        window.append(t)
        parties[counterparty(t)] += 1
        total += t.amount
        while window and t.ts - window[0].ts > span:
            old = window.popleft()
            total -= old.amount
            parties[counterparty(old)] -= 1
            if parties[counterparty(old)] == 0:
                del parties[counterparty(old)]
        if len(parties) >= min_parties and total >= min_total and (best is None or total > best[1]):
            best = (list(window), total)
    return best


def detect_fan_in(g: TransactionGraph, cfg: DetectionConfig) -> list[Finding]:
    """Smurfing: many senders -> one collector in a short window, which then forwards the pot on."""
    findings = []
    span = cfg.fan_window_hours * HOUR
    for acct, ins in g.in_txns.items():
        if len({t.sender for t in ins}) < cfg.fan_min_counterparties:
            continue
        best = _best_window(ins, lambda t: t.sender, span, cfg.fan_min_counterparties, cfg.fan_min_total)
        if best is None:
            continue
        window, total = best
        senders = {t.sender for t in window}
        start, end = window[0].ts, window[-1].ts
        # Only material outflows count as forwarding; everyday card spend is not "moving the pot".
        forwards = [t for t in g.outgoing_between(acct, start, end + cfg.fan_forward_days * DAY)
                    if t.receiver not in senders and t.amount >= 0.05 * total]
        ratio = sum(t.amount for t in forwards) / total
        if ratio < cfg.fan_in_min_forward_ratio:
            continue
        band = sum(cfg.structuring_low <= t.amount < cfg.structuring_high for t in window) / len(window)
        roles = {s: "mule" for s in senders}
        roles[acct] = "collector"
        roles.update({t.receiver: "beneficiary" for t in forwards})
        findings.append(Finding(
            detector="fan_in", source="graph",
            accounts=[acct, *sorted(senders), *sorted({t.receiver for t in forwards})],
            txn_ids=sorted({t.txn_id for t in window} | {t.txn_id for t in forwards}),
            strength=min(1.0, 0.5 + 0.3 * band + 0.2 * min(ratio, 1.0)),
            details={"collector": acct, "distinct_senders": len(senders), "window_total": round(total, 2),
                     "window_hours": round((end - start) / HOUR, 1), "forwarded_ratio": round(ratio, 3),
                     "sub_threshold_share": round(band, 3)},
            roles=roles,
        ))
    return findings


def detect_fan_out(g: TransactionGraph, cfg: DetectionConfig) -> list[Finding]:
    """One source splits funds across many receivers. Escalates to scatter-gather when the
    receivers re-converge on a common sink; plain fan-out needs sub-threshold amounts."""
    findings = []
    span = cfg.fan_window_hours * HOUR
    for acct, outs in g.out_txns.items():
        if len({t.receiver for t in outs}) < cfg.fan_min_counterparties:
            continue
        best = _best_window(outs, lambda t: t.receiver, span, cfg.fan_min_counterparties, cfg.fan_min_total)
        if best is None:
            continue
        window, total = best
        received = defaultdict(float)
        for t in window:
            received[t.receiver] += t.amount
        receivers = set(received)

        # Gather step: which downstream account receives most of each receiver's money?
        sink_votes: Counter = Counter()
        voters: dict[str, set[str]] = defaultdict(set)
        onward: dict[str, list[Txn]] = defaultdict(list)
        for r in receivers:
            first_in = min(t.ts for t in window if t.receiver == r)
            per_sink = defaultdict(list)
            for t in g.outgoing_between(r, first_in, first_in + cfg.fan_forward_days * DAY):
                if t.receiver != acct:
                    per_sink[t.receiver].append(t)
            for sink, txs in per_sink.items():
                if sum(t.amount for t in txs) >= 0.5 * received[r]:
                    sink_votes[sink] += 1
                    voters[sink].add(r)
                    onward[sink].extend(txs)

        band = sum(cfg.structuring_low <= t.amount < cfg.structuring_high for t in window) / len(window)
        sink, votes = sink_votes.most_common(1)[0] if sink_votes else (None, 0)
        gather_ratio = votes / len(receivers)

        if sink and gather_ratio >= cfg.gather_min_ratio:
            # Keep only the intermediaries that actually forwarded to the sink.
            mules = voters[sink]
            split = [t for t in window if t.receiver in mules]
            split_total = sum(t.amount for t in split)
            # Who funded the split? Largest inflow to the source shortly before it.
            funding = [t for t in g.in_txns.get(acct, [])
                       if split[0].ts - 3 * DAY <= t.ts <= split[0].ts and t.amount >= 0.5 * split_total]
            funder_txns = sorted(funding, key=lambda t: -t.amount)[:1]
            roles = {m: "mule" for m in mules}
            roles[acct], roles[sink] = "source", "sink"
            for t in funder_txns:
                roles[t.sender] = "funder"
            accounts = [*(t.sender for t in funder_txns), acct, *sorted(mules), sink]
            txn_ids = {t.txn_id for t in split} | {t.txn_id for t in onward[sink]} | {t.txn_id for t in funder_txns}
            findings.append(Finding(
                detector="scatter_gather", source="graph", accounts=accounts, txn_ids=sorted(txn_ids),
                strength=min(1.0, 0.6 + 0.4 * gather_ratio),
                details={"source": acct, "sink": sink, "n_intermediaries": len(mules),
                         "split_total": round(split_total, 2), "gather_ratio": round(gather_ratio, 3),
                         "gathered_amount": round(sum(t.amount for t in onward[sink]), 2),
                         "sub_threshold_share": round(band, 3)},
                roles=roles,
            ))
        elif band >= 0.5:
            roles = {r: "mule" for r in receivers}
            roles[acct] = "source"
            findings.append(Finding(
                detector="fan_out", source="graph", accounts=[acct, *sorted(receivers)],
                txn_ids=sorted(t.txn_id for t in window),
                strength=min(1.0, 0.5 + 0.5 * band),
                details={"source": acct, "distinct_receivers": len(receivers), "window_total": round(total, 2),
                         "sub_threshold_share": round(band, 3)},
                roles=roles,
            ))
    return findings


GRAPH_DETECTORS = (detect_cycles, detect_layering_chains, detect_fan_in, detect_fan_out)


def run_graph_detectors(g: TransactionGraph, cfg: DetectionConfig) -> list[Finding]:
    findings = [f for detector in GRAPH_DETECTORS for f in detector(g, cfg)]
    # A chain that only walks around an already-detected cycle adds no new information.
    cycles = [set(f.accounts) for f in findings if f.detector == "cycle"]
    return [f for f in findings
            if not (f.detector == "layering_chain" and any(set(f.accounts) <= c for c in cycles))]
