"""Synthetic banking data with labelled money-laundering rings.

Normal activity (retail spend, payroll, rent, P2P, B2B, remittances) is mixed with
four planted typologies so the detectors can be evaluated against ground truth:

* ``cycle``           round-tripping: funds loop through 3-6 accounts back to the origin
* ``fan_in``          smurfing: many mules deposit just under $10k into a fresh collector,
                      which forwards the pot offshore
* ``scatter_gather``  a source splits a large sum across mules who all forward to one sink
* ``layering_chain``  funds hop rapidly along a chain of accounts, shaving a little each hop

It also plants legitimate *look-alikes* (labelled clean) so detectors face realistic false
positives: home purchases through escrow (chain-shaped), families pooling a down payment
(fan-in-shaped) and private loans repaid within weeks (two-account cycles).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from .config import HIGH_RISK_COUNTRIES

LOW_RISK_FOREIGN = ("GB", "DE", "CA", "IN", "SG", "MX", "FR")
TYPOLOGIES = ("cycle", "fan_in", "scatter_gather", "layering_chain")


@dataclass
class SyntheticDataset:
    accounts: pd.DataFrame
    transactions: pd.DataFrame
    rings: list[dict] = field(default_factory=list)
    decoys: list[dict] = field(default_factory=list)


class _Generator:
    def __init__(self, n_accounts: int, days: int, start: datetime, seed: int):
        self.rng = random.Random(seed)
        self.n_accounts = n_accounts
        self.days = days
        self.start = start
        self.end = start + timedelta(days=days)
        self.accounts: list[dict] = []
        self.txns: list[dict] = []
        self.rings: list[dict] = []
        self.decoys: list[dict] = []
        self._used_in_rings: set[str] = set()

    # ------------------------------------------------------------------ helpers
    def _new_account(self, kind: str, *, country: str | None = None, recent: bool = False,
                     kyc: str | None = None) -> str:
        acct_id = f"ACC{len(self.accounts) + 1:05d}"
        r = self.rng.random()
        if country is None:
            country = "US" if r < 0.93 else (self.rng.choice(LOW_RISK_FOREIGN) if r < 0.98
                                             else self.rng.choice(HIGH_RISK_COUNTRIES))
        if recent:
            opened = self.start - timedelta(days=self.rng.randint(5, 60))
        else:
            opened = self.start - timedelta(days=self.rng.randint(180, 365 * 12))
        if kyc is None:
            kyc = self.rng.choices(["low", "medium", "high"], weights=[80, 17, 3])[0]
            if country in HIGH_RISK_COUNTRIES:
                kyc = "high"
        self.accounts.append({
            "account_id": acct_id,
            "account_type": kind,
            "country": country,
            "open_date": opened.date().isoformat(),
            "kyc_risk": kyc,
        })
        return acct_id

    def _ts(self, base: datetime | None = None, *, min_h: float = 0, max_h: float | None = None) -> datetime:
        if base is None:
            return self.start + timedelta(seconds=self.rng.uniform(0, self.days * 86_400))
        return base + timedelta(hours=self.rng.uniform(min_h, max_h if max_h is not None else min_h + 1))

    def _txn(self, sender: str, receiver: str, amount: float, ts: datetime, channel: str,
             typology: str | None = None, ring_id: str | None = None) -> None:
        if ts >= self.end:  # keep everything inside the observation window
            ts = self.end - timedelta(minutes=self.rng.randint(1, 600))
        self.txns.append({
            "timestamp": ts,
            "sender": sender,
            "receiver": receiver,
            "amount": round(amount, 2),
            "channel": channel,
            "is_laundering": int(typology is not None),
            "typology": typology,
            "ring_id": ring_id,
        })

    def _pick_unused(self, pool: list[str], k: int) -> list[str]:
        candidates = [a for a in pool if a not in self._used_in_rings]
        chosen = self.rng.sample(candidates, k)
        self._used_in_rings.update(chosen)
        return chosen

    # ----------------------------------------------------------- normal traffic
    def build_population(self) -> None:
        n_merchants = max(5, int(self.n_accounts * 0.05))
        n_employers = max(3, int(self.n_accounts * 0.02))
        n_individuals = self.n_accounts - n_merchants - n_employers
        self.merchants = [self._new_account("merchant", country="US") for _ in range(n_merchants)]
        self.employers = [self._new_account("business", country="US") for _ in range(n_employers)]
        self.individuals = [self._new_account("individual") for _ in range(n_individuals)]

    def normal_activity(self) -> None:
        rng = self.rng
        merchant_weights = [1 / (i + 1) for i in range(len(self.merchants))]  # Zipf-ish popularity
        employees = {e: [] for e in self.employers}
        for person in self.individuals:
            if rng.random() < 0.7:
                employees[rng.choice(self.employers)].append(person)

        # Payroll: legitimate fan-out on the 1st and 15th (a classic false-positive trap).
        for employer, staff in employees.items():
            salaries = {p: rng.uniform(1_500, 4_500) for p in staff}
            for day in (0, 14):
                if day >= self.days:
                    continue
                run = self.start + timedelta(days=day, hours=rng.uniform(8, 11))
                for person, salary in salaries.items():
                    self._txn(employer, person, salary * rng.uniform(0.98, 1.02),
                              run + timedelta(minutes=rng.randint(0, 90)), "ach")
            # B2B revenue so employers can fund payroll.
            for _ in range(max(2, len(staff) // 6)):
                self._txn(rng.choice(self.merchants), employer, rng.uniform(5_000, 40_000), self._ts(), "wire")

        friends = {p: rng.sample(self.individuals, k=rng.randint(2, 6)) for p in self.individuals}
        landlords = rng.sample(self.merchants, k=min(5, len(self.merchants)))
        for person in self.individuals:
            # Retail card spend at popular merchants.
            for _ in range(rng.randint(8, 25)):
                m = rng.choices(self.merchants, weights=merchant_weights)[0]
                self._txn(person, m, min(rng.lognormvariate(3.9, 0.9), 2_500), self._ts(), "card")
            # Rent.
            if rng.random() < 0.6:
                self._txn(person, rng.choice(landlords), rng.uniform(900, 3_000),
                          self.start + timedelta(days=rng.randint(0, 4), hours=rng.uniform(0, 23)), "ach")
            # P2P with friends.
            for _ in range(rng.randint(0, 5)):
                friend = rng.choice(friends[person])
                if friend != person:
                    self._txn(person, friend, rng.uniform(15, 800), self._ts(), "p2p")
            # Rare legitimate large transfer (car, tuition, family help).
            if rng.random() < 0.02:
                other = rng.choice(self.individuals)
                if other != person:
                    self._txn(person, other, rng.uniform(10_000, 45_000), self._ts(), "wire")
            # Small family remittances, sometimes to high-risk jurisdictions.
            if rng.random() < 0.03:
                foreign = [a["account_id"] for a in self.accounts
                           if a["country"] != "US" and a["account_type"] == "individual"]
                if foreign:
                    self._txn(person, rng.choice(foreign), rng.uniform(150, 2_000), self._ts(), "wire")

    # ------------------------------------------------------- planted typologies
    def _ring_start(self, span_days: float) -> datetime:
        latest = max(0.5, self.days - span_days)
        return self.start + timedelta(days=self.rng.uniform(0.5, latest))

    def plant_cycle(self, ring_id: str) -> None:
        rng = self.rng
        members = self._pick_unused(self.individuals + self.employers, rng.randint(3, 6))
        amount = rng.uniform(20_000, 150_000)
        t = self._ring_start(span_days=12)
        for _ in range(rng.randint(1, 2)):  # the loop may run more than once
            for i, sender in enumerate(members):
                receiver = members[(i + 1) % len(members)]
                self._txn(sender, receiver, amount, t, "wire", "cycle", ring_id)
                amount *= rng.uniform(0.95, 0.99)
                t = self._ts(t, min_h=6, max_h=60)
        self.rings.append({"ring_id": ring_id, "typology": "cycle", "accounts": members})

    def plant_fan_in(self, ring_id: str) -> None:
        rng = self.rng
        mules = self._pick_unused(self.individuals, rng.randint(8, 14))
        collector = self._new_account("individual", country="US", recent=True, kyc="medium")
        offshore = self._new_account("business", country=rng.choice(HIGH_RISK_COUNTRIES), recent=True)
        t0 = self._ring_start(span_days=8)
        total = 0.0
        last = t0
        for mule in mules:
            for _ in range(rng.choice((1, 1, 2))):
                amt = rng.uniform(9_000, 9_900)
                ts = self._ts(t0, min_h=0, max_h=48)
                self._txn(mule, collector, amt, ts, "cash_deposit", "fan_in", ring_id)
                total += amt
                last = max(last, ts)
        remaining = total * rng.uniform(0.85, 0.95)
        n_out = rng.randint(1, 3)
        t = self._ts(last, min_h=12, max_h=72)
        for _ in range(n_out):
            self._txn(collector, offshore, remaining / n_out, t, "wire", "fan_in", ring_id)
            t = self._ts(t, min_h=1, max_h=12)
        self.rings.append({"ring_id": ring_id, "typology": "fan_in",
                           "accounts": [*mules, collector, offshore]})

    def plant_scatter_gather(self, ring_id: str) -> None:
        rng = self.rng
        funder = self._new_account("business", country=rng.choice(HIGH_RISK_COUNTRIES), recent=True)
        source = self._pick_unused(self.employers + self.individuals, 1)[0]
        mules = self._pick_unused(self.individuals, rng.randint(8, 12))
        sink = self._new_account("business", country="US", recent=True, kyc="medium")
        splits = [rng.uniform(8_500, 9_900) for _ in mules]
        t = self._ring_start(span_days=10)
        self._txn(funder, source, sum(splits) * rng.uniform(1.01, 1.04), t, "wire", "scatter_gather", ring_id)
        for mule, amt in zip(mules, splits):
            ts = self._ts(t, min_h=2, max_h=30)
            self._txn(source, mule, amt, ts, "p2p", "scatter_gather", ring_id)
            self._txn(mule, sink, amt * rng.uniform(0.90, 0.98), self._ts(ts, min_h=12, max_h=120),
                      "p2p", "scatter_gather", ring_id)
        self.rings.append({"ring_id": ring_id, "typology": "scatter_gather",
                           "accounts": [funder, source, *mules, sink]})

    def plant_layering_chain(self, ring_id: str) -> None:
        rng = self.rng
        origin = self._new_account("business", country=rng.choice(HIGH_RISK_COUNTRIES), recent=True)
        hops = self._pick_unused(self.individuals + self.employers, rng.randint(3, 5))
        exit_acct = self._new_account("business", country=rng.choice(HIGH_RISK_COUNTRIES), recent=True)
        path = [origin, *hops, exit_acct]
        amount = rng.uniform(30_000, 200_000)
        t = self._ring_start(span_days=8)
        for sender, receiver in zip(path, path[1:]):
            self._txn(sender, receiver, amount, t, "wire", "layering_chain", ring_id)
            amount *= rng.uniform(0.95, 0.995)
            t = self._ts(t, min_h=2, max_h=30)
        self.rings.append({"ring_id": ring_id, "typology": "layering_chain", "accounts": path})

    # ----------------------------------------------- legitimate look-alikes
    def plant_home_purchase(self) -> None:
        """Buyer -> escrow -> seller -> mortgage lender within ~2 days: chain-shaped but legitimate."""
        rng = self.rng
        buyer, seller = self._pick_unused(self.individuals, 2)
        escrow, lender = rng.choice(self.employers), rng.choice(self.merchants)
        price = rng.uniform(250_000, 600_000)
        t = self._ring_start(span_days=4)
        self._txn(buyer, escrow, price, t, "wire")
        t = self._ts(t, min_h=4, max_h=30)
        self._txn(escrow, seller, price * rng.uniform(0.97, 0.99), t, "wire")  # less closing costs
        self._txn(seller, lender, price * rng.uniform(0.86, 0.95), self._ts(t, min_h=2, max_h=20), "wire")
        self.decoys.append({"kind": "home_purchase", "accounts": [buyer, escrow, seller, lender]})

    def plant_family_pooling(self) -> None:
        """Relatives each chip in for a down payment that goes to escrow: fan-in-shaped but legitimate."""
        rng = self.rng
        relatives = self._pick_unused(self.individuals, rng.randint(6, 9))
        buyer = self._pick_unused(self.individuals, 1)[0]
        escrow = rng.choice(self.employers)
        t0 = self._ring_start(span_days=8)
        total = 0.0
        for r in relatives:
            amt = rng.choice([rng.uniform(3_000, 8_000), rng.uniform(8_000, 9_800), 10_000.0])
            self._txn(r, buyer, amt, self._ts(t0, min_h=0, max_h=60), "p2p")
            total += amt
        self._txn(buyer, escrow, total * rng.uniform(0.9, 1.0), self._ts(t0, min_h=72, max_h=140), "wire")
        self.decoys.append({"kind": "family_pooling", "accounts": [*relatives, buyer, escrow]})

    def plant_private_loan(self) -> None:
        """A lends B a lump sum; B repays with interest within weeks: a 2-cycle but legitimate."""
        rng = self.rng
        lender, borrower = self._pick_unused(self.individuals, 2)
        amount = rng.uniform(15_000, 60_000)
        t = self._ring_start(span_days=20)
        self._txn(lender, borrower, amount, t, "wire")
        self._txn(borrower, lender, amount * rng.uniform(1.0, 1.03), self._ts(t, min_h=7 * 24, max_h=19 * 24), "wire")
        self.decoys.append({"kind": "private_loan", "accounts": [lender, borrower]})

    # -------------------------------------------------------------------- build
    def build(self, rings_per_typology: int, decoys_per_kind: int) -> SyntheticDataset:
        self.build_population()
        self.normal_activity()
        planters = {
            "cycle": self.plant_cycle,
            "fan_in": self.plant_fan_in,
            "scatter_gather": self.plant_scatter_gather,
            "layering_chain": self.plant_layering_chain,
        }
        n = 0
        for typology in TYPOLOGIES:
            for _ in range(rings_per_typology):
                n += 1
                planters[typology](f"RING{n:03d}")
        for _ in range(decoys_per_kind):
            self.plant_home_purchase()
            self.plant_family_pooling()
            self.plant_private_loan()

        accounts = pd.DataFrame(self.accounts)
        laundering = {a for ring in self.rings for a in ring["accounts"]}
        accounts["is_laundering"] = accounts["account_id"].isin(laundering).astype(int)

        txns = pd.DataFrame(self.txns).sort_values("timestamp", kind="stable").reset_index(drop=True)
        txns.insert(0, "txn_id", [f"TXN{i + 1:07d}" for i in range(len(txns))])
        txns["timestamp"] = txns["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return SyntheticDataset(accounts=accounts, transactions=txns, rings=self.rings, decoys=self.decoys)


def generate_dataset(n_accounts: int = 1500, days: int = 30, rings_per_typology: int = 3,
                     seed: int = 42, start: datetime = datetime(2026, 9, 1),
                     decoys_per_kind: int = 3) -> SyntheticDataset:
    """Generate accounts + transactions with ``rings_per_typology`` planted rings of each typology
    and ``decoys_per_kind`` legitimate look-alikes of each kind."""
    if n_accounts < 200:
        raise ValueError("n_accounts must be at least 200 to host the planted rings")
    return _Generator(n_accounts, days, start, seed).build(rings_per_typology, decoys_per_kind)
