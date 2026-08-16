"""test_news.py - offline checks for the news analyst.

No network, no LLM, no spend. Everything here runs without FIRECRAWL_API_KEY,
which is the point: the parts that must be right before a live run - the
openstocks guard, prompt rendering, spec construction, output validation - are
all checkable dry.

    python -m unittest analysts.test_news -v
    pytest analysts/test_news.py -q
"""

from __future__ import annotations

import unittest

import agent_core.tools as core_tools
from agent_core import settings
from agent_core.agent import create_agent

from analysts import news
from analysts.models import ExcludedSource, NewsItem, NewsReport


# ---------------------------------------------------------------------------
# 1. The openstocks guard
# ---------------------------------------------------------------------------


class BlockedUrlTests(unittest.TestCase):
    def test_blocks_the_host_in_every_form(self):
        for url in (
            "https://openstocks.com/hackathon",
            "https://www.openstocks.com/",
            "http://OPENSTOCKS.COM/company-hd",
            "https://api.openstocks.com/v1/consensus",
            "openstocks.com/hackathon",  # no scheme
            "https://openstocks.com.evil.net/x",  # look-alike; over-block on purpose
        ):
            with self.subTest(url=url):
                self.assertTrue(news.is_blocked_url(url), url)

    def test_allows_legitimate_sources(self):
        for url in (
            "https://ir.homedepot.com/news-releases",
            "https://www.reuters.com/business/retail-consumer/",
            "https://www.ft.com/content/abc",
            "https://www.londonstockexchange.com/news-article/HAS/x",
            # host is reuters.com - a PATH mentioning the host platform is
            # ordinary public reporting and must not be blocked.
            "https://www.reuters.com/technology/openstocks-hackathon-london/",
        ):
            with self.subTest(url=url):
                self.assertFalse(news.is_blocked_url(url), url)

    def test_url_host_parsing(self):
        self.assertEqual(news.url_host("https://www.Reuters.com/x?y=1"), "www.reuters.com")
        self.assertEqual(news.url_host("reuters.com/x"), "reuters.com")
        self.assertEqual(news.url_host(""), "")


class GuardInterceptionTests(unittest.TestCase):
    """The guard must stop the call BEFORE Firecrawl, not merely ask nicely."""

    def setUp(self):
        self._orig_search = core_tools.search_web
        self._orig_scrape = core_tools.scrape_markdown
        self._orig_installed = news._guard_installed

        self.scraped: list[str] = []
        self.searched: list[str] = []

        def fake_search(query, limit=10):
            self.searched.append(query)
            return [
                {"url": "https://www.reuters.com/a", "title": "wire", "description": ""},
                {"url": "https://openstocks.com/company-hd", "title": "host", "description": ""},
                {"url": "https://ir.homedepot.com/b", "title": "ir", "description": ""},
            ]

        def fake_scrape(url, max_length=None):
            self.scraped.append(url)
            return "content"

        core_tools.search_web = fake_search
        core_tools.scrape_markdown = fake_scrape
        news._guard_installed = False
        news.install_openstocks_guard()

    def tearDown(self):
        core_tools.search_web = self._orig_search
        core_tools.scrape_markdown = self._orig_scrape
        news._guard_installed = self._orig_installed

    def test_scrape_of_the_host_never_reaches_firecrawl(self):
        with news.blocked_log() as blocked:
            with self.assertRaises(news.OpenStocksBlocked):
                core_tools.scrape_markdown("https://openstocks.com/hackathon")
        self.assertEqual(self.scraped, [], "the blocked URL reached the scraper")
        self.assertEqual(len(blocked), 1)
        self.assertIn("openstocks.com", blocked[0]["target"])

    def test_legitimate_scrape_still_passes_through(self):
        self.assertEqual(core_tools.scrape_markdown("https://ir.homedepot.com/x"), "content")
        self.assertEqual(self.scraped, ["https://ir.homedepot.com/x"])

    def test_search_hits_on_the_host_are_dropped(self):
        with news.blocked_log() as blocked:
            results = core_tools.search_web("Home Depot Q2 guidance", limit=10)
        urls = [r["url"] for r in results]
        self.assertEqual(urls, ["https://www.reuters.com/a", "https://ir.homedepot.com/b"])
        self.assertEqual(len(blocked), 1)

    def test_a_query_naming_the_host_is_refused(self):
        with news.blocked_log() as blocked:
            with self.assertRaises(news.OpenStocksBlocked):
                core_tools.search_web("site:openstocks.com HD consensus")
        self.assertEqual(self.searched, [], "the blocked query reached the search API")
        self.assertEqual(len(blocked), 1)

    def test_guard_installation_is_idempotent(self):
        after_first = core_tools.scrape_markdown
        news.install_openstocks_guard()
        self.assertIs(core_tools.scrape_markdown, after_first, "guard double-wrapped itself")


# ---------------------------------------------------------------------------
# 2. Prompt
# ---------------------------------------------------------------------------


class PromptTests(unittest.TestCase):
    def test_renders_for_every_challenge_company(self):
        for ticker in ("HD", "ADI", "HAS", "DE"):
            with self.subTest(ticker=ticker):
                p = news.render_prompt(ticker, "2026-08-16")
                self.assertNotIn("{", p)
                self.assertNotIn("}", p)
                self.assertIn(news.COMPANIES[ticker].name, p)
                self.assertIn("2026-08-16", p)
                self.assertIn(news.CORPUS_FREEZE, p)

    def test_date_is_injected_not_hardcoded(self):
        a = news.render_prompt("HD", "2026-08-16")
        b = news.render_prompt("HD", "2025-11-02")
        self.assertIn("2026-08-16", a)
        self.assertNotIn("2026-08-16", b)
        self.assertIn("2025-11-02", b)

    def test_openstocks_prohibition_is_in_the_prompt(self):
        p = news.render_prompt("DE", "2026-08-16").lower()
        self.assertIn("openstocks.com", p)
        self.assertIn("never search", p)

    def test_source_quality_rules_are_explicit(self):
        p = news.render_prompt("ADI", "2026-08-16").lower()
        for token in ("reuters", "bloomberg", "financial times", "wall street journal"):
            self.assertIn(token, p)
        for token in ("technical analysis", "listicle", "penny-stock", "content-farm"):
            self.assertIn(token, p)
        self.assertIn("excluded_sources is not optional", p)

    def test_price_action_is_banned(self):
        p = news.render_prompt("HD", "2026-08-16").lower()
        self.assertIn("the stock rallied", p)
        self.assertIn("do not write about", p)

    def test_company_specific_macro_drivers(self):
        checks = {
            "HD": ["existing-home sales", "repair-and-remodel"],
            "ADI": ["book-to-bill", "utilisation"],
            "HAS": ["permanent placement", "white-collar"],
            "DE": ["net farm income", "order book"],
        }
        for ticker, tokens in checks.items():
            p = news.render_prompt(ticker, "2026-08-16").lower()
            for token in tokens:
                with self.subTest(ticker=ticker, token=token):
                    self.assertIn(token, p)

    def test_search_strategy_is_concrete(self):
        p = news.render_prompt("HAS", "2026-08-16")
        self.assertIn("at most 4 searches", p)
        self.assertIn("Hays plc full year 2026 results", p)

    def test_unknown_ticker_still_renders(self):
        p = news.render_prompt("XYZ", "2026-08-16", company_name="Example Corp")
        self.assertIn("Example Corp", p)
        self.assertNotIn("{", p)

    def test_template_validator_rejects_a_stray_brace(self):
        with self.assertRaises(ValueError):
            news._assert_renderable("A literal { brace slipped in.")
        with self.assertRaises(ValueError):
            news._assert_renderable("Unknown field {not_a_field}.")
        news._assert_renderable("Escaped {{ok}} and a real field {as_of}.")


# ---------------------------------------------------------------------------
# 3. Spec
# ---------------------------------------------------------------------------


class SpecTests(unittest.TestCase):
    def test_spec_shape(self):
        spec = news.build_spec("HD", "2026-08-16")
        self.assertIs(spec.result_model, NewsReport)
        self.assertTrue(spec.use_web)
        self.assertFalse(spec.allow_delegation)
        self.assertEqual(spec.tools, [], "the news analyst must have no corpus tools")
        self.assertIn("HD", spec.name)

    def test_model_config_comes_from_settings(self):
        spec = news.build_spec("ADI", "2026-08-16")
        self.assertIsNone(spec.profile)
        self.assertEqual(spec.resolved_profile, settings.llm_profile)
        self.assertEqual(spec.resolved_reasoning_effort, settings.reasoning_effort)
        self.assertEqual(spec.resolved_max_turns, settings.agent_max_turns)

    def test_fallback_is_a_labelled_report(self):
        spec = news.build_spec("DE", "2026-08-16")
        self.assertIsInstance(spec.fallback, NewsReport)
        self.assertEqual(spec.fallback.ticker, "DE")
        self.assertEqual(spec.fallback.as_of, "2026-08-16")
        self.assertEqual(spec.fallback.corpus_freeze, news.CORPUS_FREEZE)
        self.assertIn("FALLBACK", spec.fallback.notes)

    def test_agent_builds_with_web_tools_only(self):
        agent = create_agent(news.build_spec("HAS", "2026-08-16"))
        names = sorted(t.name for t in agent.tools)
        self.assertEqual(names, ["firecrawl_scrape", "firecrawl_search", "submit_result"])


# ---------------------------------------------------------------------------
# 4. Post-run validation
# ---------------------------------------------------------------------------


class PostprocessTests(unittest.TestCase):
    def test_identifying_fields_come_from_the_caller(self):
        report = NewsReport(ticker="WRONG", as_of="1999-01-01")
        out = news.postprocess(report, "HD", "2026-08-16")
        self.assertEqual(out.ticker, "HD")
        self.assertEqual(out.as_of, "2026-08-16")
        self.assertEqual(out.corpus_freeze, news.CORPUS_FREEZE)

    def test_quality_is_normalised(self):
        self.assertEqual(news.normalise_quality("HIGH"), "high")
        self.assertEqual(news.normalise_quality(" Medium "), "medium")
        self.assertEqual(news.normalise_quality("high (primary source)"), "high")
        self.assertEqual(news.normalise_quality("excellent"), "unrated")
        self.assertEqual(news.normalise_quality(""), "unrated")

    def test_an_openstocks_item_is_refiled_as_an_exclusion(self):
        report = NewsReport(
            items=[
                NewsItem(headline="ok", url="https://www.reuters.com/a", source_quality="high",
                         quality_reason="wire"),
                NewsItem(headline="host", url="https://openstocks.com/company-hd",
                         source_quality="high", quality_reason="n/a"),
            ]
        )
        out = news.postprocess(report, "HD", "2026-08-16")
        self.assertEqual([i.url for i in out.items], ["https://www.reuters.com/a"])
        self.assertTrue(
            any("openstocks.com" in e.url_or_publisher for e in out.excluded_sources)
        )
        self.assertIn("dropped as openstocks URLs", out.notes)

    def test_duplicates_collapse(self):
        u = "https://www.ft.com/content/x"
        report = NewsReport(
            items=[
                NewsItem(headline="a", url=u, source_quality="high", quality_reason="ft"),
                NewsItem(headline="a again", url=u, source_quality="high", quality_reason="ft"),
            ]
        )
        out = news.postprocess(report, "ADI", "2026-08-16")
        self.assertEqual(len(out.items), 1)
        self.assertIn("duplicate", out.notes)

    def test_guard_rejections_reach_the_visible_exclusion_list(self):
        out = news.postprocess(
            NewsReport(),
            "DE",
            "2026-08-16",
            blocked=[{"target": "https://openstocks.com/x", "reason": "scrape attempt"}],
        )
        self.assertEqual(len(out.excluded_sources), 1)
        self.assertIn("Blocked before the network call", out.excluded_sources[0].reason)

    def test_empty_exclusion_list_is_flagged(self):
        out = news.postprocess(NewsReport(), "HD", "2026-08-16")
        self.assertIn("nothing was excluded", out.notes)
        self.assertIn("no usable items", out.notes)

    def test_missing_quality_reason_is_flagged(self):
        report = NewsReport(
            items=[NewsItem(headline="a", url="https://reuters.com/a", source_quality="high")],
            excluded_sources=[ExcludedSource(url_or_publisher="somefarm.io", reason="rewrite")],
        )
        out = news.postprocess(report, "HD", "2026-08-16")
        self.assertIn("no quality_reason", out.notes)


# ---------------------------------------------------------------------------
# 5. Rendering and CLI
# ---------------------------------------------------------------------------


class SummaryTests(unittest.TestCase):
    def _sample(self) -> NewsReport:
        return news.postprocess(
            NewsReport(
                items=[
                    NewsItem(
                        headline="Home Depot reaffirms fiscal 2026 outlook",
                        url="https://ir.homedepot.com/news",
                        published="2026-08-15",
                        publisher="Home Depot IR",
                        source_quality="high",
                        quality_reason="First-party IR release.",
                        fundamental_relevance="Comparable sales, total company",
                        implication="Full-year comp range unchanged.",
                        direction="neutral",
                    ),
                    NewsItem(
                        headline="Housing turnover slips",
                        url="https://www.reuters.com/x",
                        published="2026-08-15",
                        publisher="Reuters",
                        source_quality="medium",
                        quality_reason="Wire report of NAR data, no primary release read.",
                        fundamental_relevance="Net sales",
                        implication="Big-ticket demand stays soft.",
                        direction="negative",
                    ),
                ],
                valuation_view="Earnings power intact; the cycle is the swing factor.",
                excluded_sources=[
                    ExcludedSource(url_or_publisher="stockhype.example", reason="chart-based hype"),
                ],
                confidence="medium",
            ),
            "HD",
            "2026-08-16",
        )

    def test_summary_groups_by_quality_and_shows_rejects(self):
        text = news.format_summary(self._sample())
        self.assertIn("HIGH QUALITY (1)", text)
        self.assertIn("MEDIUM QUALITY (1)", text)
        self.assertIn("VALUATION VIEW", text)
        self.assertIn("EXCLUDED SOURCES (1)", text)
        self.assertIn("stockhype.example", text)
        self.assertIn("why high:", text)

    def test_summary_of_an_empty_report_does_not_crash(self):
        self.assertIn("NEWS ANALYST", news.format_summary(NewsReport()))


class CliTests(unittest.TestCase):
    def test_cli_fails_clearly_without_a_firecrawl_key(self):
        original = settings.firecrawl_api_key
        settings.firecrawl_api_key = ""
        try:
            code = news.main(["--ticker", "HD", "--as-of", "2026-08-16"])
        finally:
            settings.firecrawl_api_key = original
        self.assertEqual(code, 2)

    def test_require_firecrawl_message_is_actionable(self):
        original = settings.firecrawl_api_key
        settings.firecrawl_api_key = ""
        try:
            with self.assertRaises(Exception) as ctx:
                news.require_firecrawl()
        finally:
            settings.firecrawl_api_key = original
        msg = str(ctx.exception)
        self.assertIn("FIRECRAWL_API_KEY", msg)
        self.assertIn(".env", msg)

    def test_print_prompt_works_without_a_key(self):
        original = settings.firecrawl_api_key
        settings.firecrawl_api_key = ""
        try:
            code = news.main(["--ticker", "DE", "--as-of", "2026-08-16", "--print-prompt"])
        finally:
            settings.firecrawl_api_key = original
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
