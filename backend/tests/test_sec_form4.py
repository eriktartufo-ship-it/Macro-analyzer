"""Test SEC EDGAR Form 4 insider trading aggregator.

Verifica:
- Parsing daily index (form.YYYYMMDD.idx) per estrarre Form 4 entries.
- Parsing Form 4 XML per estrarre transactionCode + shares + price.
- Aggregator: buy/sell ratio, net dollar flow, normalized score.
- Edge cases: filings vuoti, codici M/A esclusi, missing fields.
"""
from __future__ import annotations

from datetime import date

from app.services.alt_data.sec_form4 import (
    Form4Transaction,
    aggregate_insider_activity,
    parse_daily_index_text,
    parse_form4_xml,
)


# ---- Fixtures inline ----------------------------------------------------------

_DAILY_INDEX_SAMPLE = """Description:           Daily Index of EDGAR Dissemination Feed
Last Data Received:    May 11, 2026
Comments:              webmaster@sec.gov
Anonymous FTP:         ftp://ftp.sec.gov/edgar/

Form Type   Company Name                                                  CIK         Date Filed  Filename
---------------------------------------------------------------------------------------------------------------------------------------------
10-K        SOME COMPANY INC                                              0000001234  2026-05-11  edgar/data/1234/0001234567-26-000001-index.htm
4           APPLE INC                                                     0000320193  2026-05-11  edgar/data/320193/0000320193-26-000045-index.htm
4           MICROSOFT CORP                                                0000789019  2026-05-11  edgar/data/789019/0000789019-26-000123-index.htm
8-K         RANDOM CORP                                                   0000999999  2026-05-11  edgar/data/999999/0000999999-26-000007-index.htm
4           TESLA INC                                                     0001318605  2026-05-11  edgar/data/1318605/0001318605-26-000089-index.htm
"""


_FORM4_XML_PURCHASE = """<?xml version="1.0"?>
<ownershipDocument>
  <schemaVersion>X0306</schemaVersion>
  <documentType>4</documentType>
  <periodOfReport>2026-05-10</periodOfReport>
  <issuer>
    <issuerCik>0000320193</issuerCik>
    <issuerName>APPLE INC</issuerName>
    <issuerTradingSymbol>AAPL</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0001214156</rptOwnerCik><rptOwnerName>COOK TIMOTHY D</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isDirector>0</isDirector><isOfficer>1</isOfficer><officerTitle>CEO</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-05-10</value></transactionDate>
      <transactionCoding>
        <transactionFormType>4</transactionFormType>
        <transactionCode>P</transactionCode>
        <equitySwapInvolved>0</equitySwapInvolved>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>180.50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


_FORM4_XML_SALE = """<?xml version="1.0"?>
<ownershipDocument>
  <documentType>4</documentType>
  <issuer>
    <issuerCik>0000789019</issuerCik>
    <issuerTradingSymbol>MSFT</issuerTradingSymbol>
  </issuer>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-05-10</value></transactionDate>
      <transactionCoding>
        <transactionCode>S</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>5000</value></transactionShares>
        <transactionPricePerShare><value>420.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


_FORM4_XML_MIXED = """<?xml version="1.0"?>
<ownershipDocument>
  <documentType>4</documentType>
  <issuer><issuerCik>0001318605</issuerCik><issuerTradingSymbol>TSLA</issuerTradingSymbol></issuer>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-05-10</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>500</value></transactionShares>
        <transactionPricePerShare><value>250.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-05-10</value></transactionDate>
      <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>10000</value></transactionShares>
        <transactionPricePerShare><value>5.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-05-10</value></transactionDate>
      <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>20000</value></transactionShares>
        <transactionPricePerShare><value>0.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


class TestParseDailyIndex:
    def test_extracts_form4_entries_only(self):
        entries = parse_daily_index_text(_DAILY_INDEX_SAMPLE, filing_date=date(2026, 5, 11))
        assert len(entries) == 3
        ciks = {e.cik for e in entries}
        assert ciks == {"0000320193", "0000789019", "0001318605"}

    def test_extracts_filename_path(self):
        entries = parse_daily_index_text(_DAILY_INDEX_SAMPLE, filing_date=date(2026, 5, 11))
        apple = next(e for e in entries if e.cik == "0000320193")
        assert "edgar/data/320193" in apple.filename
        assert apple.company == "APPLE INC"

    def test_empty_index_returns_empty_list(self):
        empty = "Form Type   Company Name   CIK   Date Filed   Filename\n----\n"
        assert parse_daily_index_text(empty, filing_date=date(2026, 5, 11)) == []

    def test_no_form4_returns_empty(self):
        text = """Form Type   Company Name   CIK   Date Filed   Filename
----
10-K   FOO   0000111111   2026-05-11   edgar/data/111111/foo-index.htm
8-K    BAR   0000222222   2026-05-11   edgar/data/222222/bar-index.htm
"""
        assert parse_daily_index_text(text, filing_date=date(2026, 5, 11)) == []


class TestParseForm4Xml:
    def test_parse_purchase_extracts_pos_signed_shares(self):
        tx = parse_form4_xml(_FORM4_XML_PURCHASE)
        assert len(tx) == 1
        t = tx[0]
        assert t.code == "P"
        assert t.shares == 1000
        assert t.price == 180.5
        assert t.acquired_disposed == "A"
        assert t.ticker == "AAPL"
        assert t.dollar_value() == 180500.0

    def test_parse_sale_extracts_disposed(self):
        tx = parse_form4_xml(_FORM4_XML_SALE)
        assert len(tx) == 1
        t = tx[0]
        assert t.code == "S"
        assert t.acquired_disposed == "D"
        assert t.dollar_value() == 5000 * 420.0

    def test_parse_mixed_returns_all_transactions(self):
        """Tutte le transactions estratte; filtering codici è step successivo."""
        tx = parse_form4_xml(_FORM4_XML_MIXED)
        assert len(tx) == 3
        codes = {t.code for t in tx}
        assert codes == {"P", "M", "A"}

    def test_parse_malformed_returns_empty(self):
        assert parse_form4_xml("<not xml>") == []
        assert parse_form4_xml("") == []


class TestAggregateInsiderActivity:
    def _tx(self, code: str, shares: float, price: float, ticker: str = "AAPL"):
        return Form4Transaction(
            code=code,
            shares=shares,
            price=price,
            acquired_disposed="A" if code in {"P", "M", "A"} else "D",
            ticker=ticker,
            transaction_date=date(2026, 5, 10),
        )

    def test_open_market_purchases_sum_to_buy_total(self):
        txs = [
            self._tx("P", 100, 50.0),
            self._tx("P", 200, 50.0),
        ]
        agg = aggregate_insider_activity(txs)
        assert agg.total_buy_usd == 15000.0
        assert agg.total_sell_usd == 0.0
        assert agg.n_buy_filings == 2
        assert agg.n_sell_filings == 0

    def test_open_market_sales_sum_to_sell_total(self):
        txs = [
            self._tx("S", 1000, 100.0),
            self._tx("S", 500, 200.0),
        ]
        agg = aggregate_insider_activity(txs)
        assert agg.total_sell_usd == 200000.0
        assert agg.total_buy_usd == 0.0

    def test_excludes_option_exercises_and_grants(self):
        """Codici M (exercise) e A (grant) NON sono market signal — escludi."""
        txs = [
            self._tx("P", 100, 50.0),   # buy = 5000
            self._tx("M", 10000, 1.0),  # exclude (option exercise)
            self._tx("A", 5000, 0.0),   # exclude (compensation grant)
            self._tx("S", 200, 100.0),  # sell = 20000
        ]
        agg = aggregate_insider_activity(txs)
        assert agg.total_buy_usd == 5000.0
        assert agg.total_sell_usd == 20000.0

    def test_buy_sell_ratio_and_net_flow(self):
        txs = [
            self._tx("P", 1000, 100.0),  # buy 100k
            self._tx("S", 500, 100.0),   # sell 50k
        ]
        agg = aggregate_insider_activity(txs)
        assert agg.total_buy_usd == 100_000.0
        assert agg.total_sell_usd == 50_000.0
        assert agg.buy_sell_ratio == 2.0
        assert agg.net_flow_usd == 50_000.0

    def test_score_positive_when_buying_dominates(self):
        # $200M net buy → ~tanh(2.0) ≈ 0.96 (saturato S&P 500 daily)
        txs = [self._tx("P", 1_000_000, 200.0)]
        agg = aggregate_insider_activity(txs)
        assert agg.insider_score > 0.5

    def test_score_negative_when_selling_dominates(self):
        # $200M net sell
        txs = [self._tx("S", 1_000_000, 200.0)]
        agg = aggregate_insider_activity(txs)
        assert agg.insider_score < -0.5

    def test_score_zero_when_no_transactions(self):
        agg = aggregate_insider_activity([])
        assert agg.insider_score == 0.0
        assert agg.total_buy_usd == 0.0
        assert agg.total_sell_usd == 0.0
        assert agg.buy_sell_ratio is None

    def test_score_in_bounded_range(self):
        # Anche net flow enorme, score limitato a [-1, +1]
        txs = [self._tx("P", 10_000_000, 1000.0)]
        agg = aggregate_insider_activity(txs)
        assert -1.0 <= agg.insider_score <= 1.0
