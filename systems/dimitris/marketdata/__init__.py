"""marketdata — market prices and analyst consensus, the two things the corpus lacks.

The offline corpus (`corpus/`) is the primary evidence base and holds every
filing we need. This package deliberately fetches NO filings. It supplies only
what a frozen document set cannot: live quotes, price history, and the sell-side
consensus / revision context that anchors a forecast against the market.
"""

from __future__ import annotations
