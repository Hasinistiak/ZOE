from __future__ import annotations

import asyncio
import html
import os
import re
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY", "").strip()

NEWSDATA_URL = "https://newsdata.io/api/1/latest"

REQUEST_TIMEOUT = 10


# ============================================================
# FRESHNESS
# ============================================================

# /latest from NewsData is already intended for recent news.
# ZOE additionally enforces this independently.
NEWS_MAX_AGE_HOURS = 48

# Allow small clock differences between ZOE and publishers.
FUTURE_ARTICLE_TOLERANCE_MINUTES = 15


# ============================================================
# GOOGLE NEWS RESOLUTION
# ============================================================

GOOGLE_RESOLVE_TIMEOUT = 12
GOOGLE_RESOLVE_CONCURRENCY = 8


# ============================================================
# CACHE
# ============================================================

CACHE_TTL = 120
URL_RESOLUTION_CACHE_TTL = 60 * 60 * 6

DEFAULT_LIMIT = 8
MAX_LIMIT = 20

# NewsData allows up to 10 articles/request on the relevant tier.
NEWSDATA_PAGE_SIZE = 10


# ============================================================
# USER AGENT
# ============================================================

BROWSER_USER_AGENT = (
    "Mozilla/5.0 "
    "(X11; Linux x86_64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/140.0.0.0 "
    "Safari/537.36"
)


# ============================================================
# CACHES
# ============================================================

_cache: dict[str, tuple[float, Any]] = {}

_url_resolution_cache: dict[
    str,
    tuple[float, str],
] = {}


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class NewsArticle:
    title: str
    description: str
    source: str
    url: str
    published_at: str
    category: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# CACHE HELPERS
# ============================================================

def _cache_get(key: str) -> Any | None:
    cached = _cache.get(key)

    if cached is None:
        return None

    timestamp, value = cached

    if time.time() - timestamp >= CACHE_TTL:
        _cache.pop(key, None)
        return None

    return value


def _cache_set(key: str, value: Any) -> None:
    _cache[key] = (
        time.time(),
        value,
    )


# ============================================================
# URL RESOLUTION CACHE
# ============================================================

def _resolved_url_cache_get(url: str) -> str | None:
    key = url.strip()

    cached = _url_resolution_cache.get(key)

    if cached is None:
        return None

    timestamp, value = cached

    if time.time() - timestamp >= URL_RESOLUTION_CACHE_TTL:
        _url_resolution_cache.pop(key, None)
        return None

    return value


def _resolved_url_cache_set(
    url: str,
    resolved_url: str,
) -> None:
    source = url.strip()
    destination = resolved_url.strip()

    if not source or not destination:
        return

    if _is_google_news_url(destination):
        return

    _url_resolution_cache[source] = (
        time.time(),
        destination,
    )


# ============================================================
# SAFE ERROR LOGGING
# ============================================================

def _redact_sensitive_text(value: Any) -> str:
    text = str(value)

    # API key in URL query string.
    text = re.sub(
        r"([?&]apikey=)[^&\s]+",
        r"\1<redacted>",
        text,
        flags=re.IGNORECASE,
    )

    # API-key-like JSON / text fields.
    text = re.sub(
        r'(["\']?(?:apikey|api_key|key)["\']?\s*[:=]\s*)'
        r'(["\']?)[^,"\'}\s]+',
        r"\1<redacted>",
        text,
        flags=re.IGNORECASE,
    )

    return text


# ============================================================
# TEXT HELPERS
# ============================================================

def _clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = html.unescape(str(value))

    text = re.sub(
        r"<[^>]*>",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _clean_url(value: Any) -> str:
    if not value:
        return ""

    return str(value).strip()


def _normalize_title(title: str) -> str:
    title = title.lower()

    title = re.sub(
        r"[^a-z0-9\s]",
        " ",
        title,
    )

    title = re.sub(
        r"\s+",
        " ",
        title,
    )

    return title.strip()


# ============================================================
# URL HELPERS
# ============================================================

def _normalize_url(url: str) -> str:
    value = str(url or "").strip()

    if not value:
        return ""

    scheme_match = re.match(
        r"^([a-zA-Z][a-zA-Z0-9+.-]*):",
        value,
    )

    if scheme_match:
        scheme = scheme_match.group(1).lower()

        if scheme not in {
            "http",
            "https",
        }:
            return ""

    if not re.match(
        r"^https?://",
        value,
        re.IGNORECASE,
    ):
        value = "https://" + value

    return value


def _is_google_news_url(url: str) -> bool:
    value = _normalize_url(url)

    if not value:
        return False

    try:
        parsed = urllib.parse.urlparse(value)

        hostname = (
            parsed.hostname or ""
        ).lower()

        return (
            hostname == "news.google.com"
            or hostname.endswith(".news.google.com")
        )

    except Exception:
        return False


def _is_valid_publisher_url(url: str) -> bool:
    value = _normalize_url(url)

    if not value:
        return False

    if _is_google_news_url(value):
        return False

    try:
        parsed = urllib.parse.urlparse(value)

        return (
            parsed.scheme in {
                "http",
                "https",
            }
            and bool(parsed.hostname)
        )

    except Exception:
        return False


# ============================================================
# DATE HANDLING
# ============================================================

def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None

    text = _clean_text(value)

    if not text:
        return None

    # --------------------------------------------------------
    # ISO-8601
    # --------------------------------------------------------

    try:
        normalized = text

        if normalized.endswith("Z"):
            normalized = (
                normalized[:-1]
                + "+00:00"
            )

        dt = datetime.fromisoformat(normalized)

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt.astimezone(timezone.utc)

    except ValueError:
        pass

    # --------------------------------------------------------
    # RFC-822 / RSS
    # --------------------------------------------------------

    try:
        dt = parsedate_to_datetime(text)

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt.astimezone(timezone.utc)

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        pass

    # --------------------------------------------------------
    # A few common provider formats.
    # --------------------------------------------------------

    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    )

    for fmt in formats:
        try:
            dt = datetime.strptime(
                text,
                fmt,
            )

            return dt.replace(
                tzinfo=timezone.utc
            )

        except ValueError:
            continue

    return None


def _normalize_published_at(value: Any) -> str:
    dt = _parse_datetime(value)

    if dt is None:
        return ""

    return (
        dt.astimezone(timezone.utc)
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


def _is_fresh_article(
    published_at: str,
) -> bool:
    dt = _parse_datetime(published_at)

    if dt is None:
        return False

    now = datetime.now(timezone.utc)

    max_age = timedelta(
        hours=NEWS_MAX_AGE_HOURS
    )

    future_tolerance = timedelta(
        minutes=FUTURE_ARTICLE_TOLERANCE_MINUTES
    )

    # Reject suspicious future timestamps.
    if dt > now + future_tolerance:
        return False

    # Reject old articles.
    if now - dt > max_age:
        return False

    return True


def _filter_fresh_articles(
    articles: list[NewsArticle],
) -> list[NewsArticle]:

    fresh: list[NewsArticle] = []

    for article in articles:

        normalized_date = _normalize_published_at(
            article.published_at
        )

        # For latest-news retrieval, undated articles
        # are not acceptable.
        if not normalized_date:
            continue

        article.published_at = normalized_date

        if not _is_fresh_article(
            normalized_date
        ):
            continue

        fresh.append(article)

    fresh.sort(
        key=lambda article: (
            _parse_datetime(
                article.published_at
            )
            or datetime.min.replace(
                tzinfo=timezone.utc
            )
        ),
        reverse=True,
    )

    return fresh


# ============================================================
# URL CANONICALIZATION
# ============================================================

def _canonicalize_url(url: str) -> str:
    normalized = _normalize_url(url)

    if not normalized:
        return ""

    try:
        parsed = urllib.parse.urlparse(
            normalized
        )

        hostname = (
            parsed.hostname or ""
        ).lower()

        query_pairs = urllib.parse.parse_qsl(
            parsed.query,
            keep_blank_values=True,
        )

        tracking_prefixes = (
            "utm_",
            "fbclid",
            "gclid",
            "mc_",
            "ref_",
        )

        filtered_pairs = [
            (
                key,
                value,
            )
            for key, value in query_pairs
            if not key.lower().startswith(
                tracking_prefixes
            )
        ]

        clean_query = urllib.parse.urlencode(
            filtered_pairs
        )

        return urllib.parse.urlunparse(
            (
                parsed.scheme.lower(),
                hostname,
                parsed.path.rstrip("/"),
                "",
                clean_query,
                "",
            )
        ).lower()

    except Exception:
        return normalized.lower()


# ============================================================
# DEDUPLICATION
# ============================================================

def _deduplicate(
    articles: list[NewsArticle],
) -> list[NewsArticle]:

    seen_urls: set[str] = set()

    seen_title_source: set[
        tuple[str, str]
    ] = set()

    result: list[NewsArticle] = []

    for article in articles:

        url_key = _canonicalize_url(
            article.url
        )

        title_key = _normalize_title(
            article.title
        )

        source_key = _normalize_title(
            article.source
        )

        if not title_key:
            continue

        # Same canonical URL.
        if (
            url_key
            and url_key in seen_urls
        ):
            continue

        # Same title from same publisher.
        title_source_key = (
            title_key,
            source_key,
        )

        if (
            title_source_key
            in seen_title_source
        ):
            continue

        if url_key:
            seen_urls.add(url_key)

        seen_title_source.add(
            title_source_key
        )

        result.append(article)

    return result


# ============================================================
# GOOGLE NEWS URL RESOLUTION
# ============================================================

async def _resolve_google_article(
    url: str,
    browser,
    semaphore: asyncio.Semaphore,
) -> str | None:

    url = _normalize_url(url)

    if not url:
        return None

    if not _is_google_news_url(url):
        return url

    cached = _resolved_url_cache_get(url)

    if (
        cached
        and _is_valid_publisher_url(cached)
    ):
        return cached

    async with semaphore:

        page = None

        try:
            page = await browser.new_page(
                user_agent=BROWSER_USER_AGENT,
                viewport={
                    "width": 1280,
                    "height": 900,
                },
                locale="en-US",
            )

            # ------------------------------------------------
            # Avoid unnecessary resources.
            # ------------------------------------------------

            try:
                await page.route(
                    "**/*",
                    lambda route: (
                        route.abort()
                        if route.request.resource_type
                        in {
                            "image",
                            "media",
                            "font",
                        }
                        else route.continue_()
                    ),
                )

            except Exception:
                pass

            # ------------------------------------------------
            # Navigate.
            # ------------------------------------------------

            try:
                await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=(
                        GOOGLE_RESOLVE_TIMEOUT
                        * 1000
                    ),
                )

            except Exception:
                return None

            try:
                await page.wait_for_timeout(
                    250
                )
            except Exception:
                pass

            final_url = _normalize_url(
                page.url
            )

            # ------------------------------------------------
            # Best case: Google redirected directly.
            # ------------------------------------------------

            if _is_valid_publisher_url(
                final_url
            ):
                _resolved_url_cache_set(
                    url,
                    final_url,
                )

                return final_url

            # ------------------------------------------------
            # Meta refresh.
            # ------------------------------------------------

            try:
                meta_url = await page.evaluate(
                    """
                    () => {
                        const meta =
                            document.querySelector(
                                'meta[http-equiv="refresh"]'
                            );

                        if (!meta) return "";

                        const content =
                            meta.getAttribute("content") || "";

                        const match =
                            content.match(
                                /url\\s*=\\s*(.+)$/i
                            );

                        return match
                            ? match[1].trim()
                            : "";
                    }
                    """
                )

                meta_url = _normalize_url(
                    meta_url
                )

                if _is_valid_publisher_url(
                    meta_url
                ):
                    _resolved_url_cache_set(
                        url,
                        meta_url,
                    )

                    return meta_url

            except Exception:
                pass

            # ------------------------------------------------
            # Carefully inspect likely article links.
            # ------------------------------------------------

            try:
                candidate_links = await page.locator(
                    "article a[href], "
                    "main a[href], "
                    "a[href][target='_blank']"
                ).evaluate_all(
                    """
                    elements =>
                        elements
                            .map(element => ({
                                href: element.href,
                                text: (
                                    element.innerText || ""
                                ).trim()
                            }))
                            .filter(item => item.href)
                    """
                )

                for candidate in candidate_links:

                    candidate_url = _normalize_url(
                        candidate.get(
                            "href",
                            "",
                        )
                    )

                    candidate_text = _clean_text(
                        candidate.get(
                            "text",
                            "",
                        )
                    )

                    if not _is_valid_publisher_url(
                        candidate_url
                    ):
                        continue

                    parsed = urllib.parse.urlparse(
                        candidate_url
                    )

                    path = (
                        parsed.path or ""
                    ).strip("/")

                    if (
                        not candidate_text
                        and not path
                    ):
                        continue

                    _resolved_url_cache_set(
                        url,
                        candidate_url,
                    )

                    return candidate_url

            except Exception:
                pass

            return None

        except Exception:
            return None

        finally:

            if page is not None:
                try:
                    await page.close()
                except Exception:
                    pass


async def _resolve_google_article_with_timeout(
    url: str,
    browser,
    semaphore: asyncio.Semaphore,
) -> str | None:

    try:
        return await asyncio.wait_for(
            _resolve_google_article(
                url,
                browser,
                semaphore,
            ),
            timeout=GOOGLE_RESOLVE_TIMEOUT,
        )

    except asyncio.TimeoutError:
        return None


async def _resolve_google_urls_async(
    urls: list[str],
) -> dict[str, str]:

    if not urls:
        return {}

    unique_urls: list[str] = []
    seen: set[str] = set()

    for raw_url in urls:

        url = _normalize_url(raw_url)

        if not url:
            continue

        if not _is_google_news_url(url):
            continue

        if url in seen:
            continue

        seen.add(url)
        unique_urls.append(url)

    if not unique_urls:
        return {}

    results: dict[str, str] = {}
    unresolved: list[str] = []

    # --------------------------------------------------------
    # Cache.
    # --------------------------------------------------------

    for url in unique_urls:

        cached = _resolved_url_cache_get(
            url
        )

        if (
            cached
            and _is_valid_publisher_url(cached)
        ):
            results[url] = cached

        else:
            unresolved.append(url)

    if not unresolved:
        return results

    # --------------------------------------------------------
    # Playwright.
    # --------------------------------------------------------

    try:
        from playwright.async_api import (
            async_playwright,
        )

    except ImportError:
        print(
            "[ZOE NEWS] "
            "Playwright is not installed. "
            "Google URLs will be kept."
        )

        return results

    semaphore = asyncio.Semaphore(
        GOOGLE_RESOLVE_CONCURRENCY
    )

    try:

        async with async_playwright() as playwright:

            browser = await playwright.chromium.launch(
                headless=True,
            )

            try:

                tasks = [
                    _resolve_google_article_with_timeout(
                        url,
                        browser,
                        semaphore,
                    )
                    for url in unresolved
                ]

                resolved = await asyncio.gather(
                    *tasks,
                    return_exceptions=True,
                )

                for (
                    source_url,
                    destination,
                ) in zip(
                    unresolved,
                    resolved,
                ):

                    if isinstance(
                        destination,
                        Exception,
                    ):
                        continue

                    if not destination:
                        continue

                    destination = _normalize_url(
                        destination
                    )

                    if not _is_valid_publisher_url(
                        destination
                    ):
                        continue

                    results[
                        source_url
                    ] = destination

            finally:

                try:
                    await browser.close()
                except Exception:
                    pass

    except Exception as exc:

        print(
            "[ZOE NEWS] "
            "Google URL resolution unavailable:",
            _redact_sensitive_text(exc),
        )

    return results


def _run_async_in_thread(
    coroutine,
) -> Any:

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:

        try:
            result["value"] = asyncio.run(
                coroutine
            )

        except BaseException as exc:
            error["value"] = exc

    thread = threading.Thread(
        target=runner,
        daemon=True,
    )

    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]

    return result.get("value")


def _resolve_google_urls(
    urls: list[str],
) -> dict[str, str]:

    if not urls:
        return {}

    try:

        return _run_async_in_thread(
            _resolve_google_urls_async(urls)
        )

    except Exception as exc:

        print(
            "[ZOE NEWS] "
            "Could not resolve Google URLs:",
            _redact_sensitive_text(exc),
        )

        return {}


# ============================================================
# PUBLIC URL RESOLUTION
# ============================================================

def resolve_article_url(
    url: str,
) -> str:

    normalized = _normalize_url(url)

    if not normalized:
        return ""

    if not _is_google_news_url(normalized):
        return normalized

    resolved = _resolve_google_urls(
        [normalized]
    )

    return resolved.get(
        normalized,
        # IMPORTANT:
        # If Google resolution fails, keep the original
        # Google News URL instead of returning nothing.
        normalized,
    )


# ============================================================
# ARTICLE URL PROCESSING
# ============================================================

def _resolve_article_urls(
    articles: list[NewsArticle],
) -> list[NewsArticle]:

    if not articles:
        return []

    google_urls: list[str] = []

    for article in articles:

        article.url = _normalize_url(
            article.url
        )

        if _is_google_news_url(
            article.url
        ):
            google_urls.append(
                article.url
            )

    if not google_urls:
        return articles

    resolved_map = _resolve_google_urls(
        google_urls
    )

    result: list[NewsArticle] = []

    for article in articles:

        url = article.url

        if not url:
            continue

        # ----------------------------------------------------
        # Direct publisher URL.
        # ----------------------------------------------------

        if not _is_google_news_url(url):

            if _is_valid_publisher_url(url):
                result.append(article)

            continue

        # ----------------------------------------------------
        # Google News URL.
        #
        # CRITICAL:
        #
        # If resolution fails, DO NOT THROW THE ARTICLE AWAY.
        # Keep the Google News URL.
        # ----------------------------------------------------

        resolved = resolved_map.get(url)

        if resolved and _is_valid_publisher_url(
            resolved
        ):
            article.url = resolved

        # Otherwise article.url remains the Google URL.

        result.append(article)

    return result


# ============================================================
# LIMIT
# ============================================================

def _normalize_limit(
    limit: int,
) -> int:

    try:
        value = int(limit)

    except (
        TypeError,
        ValueError,
    ):
        return DEFAULT_LIMIT

    return max(
        1,
        min(
            value,
            MAX_LIMIT,
        ),
    )


# ============================================================
# QUERY NORMALIZATION
# ============================================================

def _normalize_search_query(
    query: str,
) -> str:

    query = _clean_text(query)

    if not query:
        return ""

    # Common voice/STT spelling mistakes.
    replacements = {
        "lateest": "latest",
        "latset": "latest",
        "lates": "latest",
        "latestt": "latest",
        "todays": "today",
        "modle": "model",
        "modles": "models",
    }

    words = query.split()

    normalized_words: list[str] = []

    for word in words:

        replacement = replacements.get(
            word.lower()
        )

        if replacement:
            normalized_words.append(
                replacement
            )
        else:
            normalized_words.append(word)

    return " ".join(
        normalized_words
    ).strip()


def _remove_freshness_words(
    query: str,
) -> str:

    query = _normalize_search_query(
        query
    )

    if not query:
        return ""

    freshness_words = {
        "latest",
        "recent",
        "today",
        "new",
        "newest",
        "breaking",
        "current",
        "currently",
        "just",
        "recently",
    }

    words = query.split()

    remaining: list[str] = []

    for word in words:

        if word.lower() in freshness_words:
            continue

        remaining.append(word)

    return " ".join(
        remaining
    ).strip()


# ============================================================
# NEWSDATA
# ============================================================

def _fetch_newsdata(
    *,
    query: str | None,
    category: str,
    limit: int,
) -> list[NewsArticle]:

    if not NEWSDATA_API_KEY:
        print(
            "[ZOE NEWS] "
            "NewsData API key is not configured."
        )
        return []

    limit = _normalize_limit(limit)

    clean_query = ""

    if query:
        clean_query = _normalize_search_query(
            query
        )

    params: dict[str, str] = {
        "apikey": NEWSDATA_API_KEY,
        "language": "en",
        "removeduplicate": "1",
        "size": str(
            min(
                NEWSDATA_PAGE_SIZE,
                limit,
            )
        ),
    }

    # ========================================================
    # IMPORTANT:
    #
    # Explicit queries are GLOBAL searches.
    #
    # Do NOT add category=world when the user asked for
    # something specific such as "best AI model".
    # ========================================================

    if clean_query:

        params["q"] = clean_query

    else:

        if category in {
            "football",
            "f1",
        }:
            params["category"] = "sports"

        elif category == "world":
            params["category"] = "world"

    cache_key = (
        "newsdata:"
        + category
        + ":"
        + str(limit)
        + ":"
        + urllib.parse.urlencode(
            sorted(
                {
                    key: value
                    for key, value in params.items()
                    if key != "apikey"
                }.items()
            )
        )
    )

    cached = _cache_get(cache_key)

    if cached is not None:
        return cached[:limit]

    articles: list[NewsArticle] = []

    next_page: str | None = None

    max_pages = (
        (limit // NEWSDATA_PAGE_SIZE)
        + 3
    )

    for _ in range(max_pages):

        request_params = dict(params)

        if next_page:
            request_params["page"] = next_page

        try:

            response = requests.get(
                NEWSDATA_URL,
                params=request_params,
                timeout=REQUEST_TIMEOUT,
            )

        except requests.RequestException as exc:

            print(
                "[ZOE NEWS] "
                "NewsData request failed:",
                _redact_sensitive_text(exc),
            )

            break

        # ----------------------------------------------------
        # API error.
        # ----------------------------------------------------

        if not response.ok:

            try:
                error_payload = response.json()

            except ValueError:
                error_payload = (
                    response.text[:1000]
                )

            print(
                "[ZOE NEWS] "
                f"NewsData HTTP "
                f"{response.status_code}: "
                f"{_redact_sensitive_text(error_payload)}"
            )

            break

        # ----------------------------------------------------
        # JSON.
        # ----------------------------------------------------

        try:

            payload = response.json()

        except ValueError as exc:

            print(
                "[ZOE NEWS] "
                "NewsData returned invalid JSON:",
                repr(exc),
            )

            break

        if payload.get("status") != "success":

            print(
                "[ZOE NEWS] "
                "NewsData unsuccessful response:",
                _redact_sensitive_text(payload),
            )

            break

        results = payload.get(
            "results",
            [],
        )

        # ----------------------------------------------------
        # Parse.
        # ----------------------------------------------------

        for item in results:

            title = _clean_text(
                item.get("title")
            )

            if not title:
                continue

            description = (
                _clean_text(
                    item.get("description")
                )
                or _clean_text(
                    item.get("content")
                )
            )

            source = (
                _clean_text(
                    item.get("source_name")
                )
                or _clean_text(
                    item.get("source_id")
                )
                or "Unknown source"
            )

            url = _clean_url(
                item.get("link")
            )

            published_at = (
                _clean_text(
                    item.get("pubDate")
                )
                or _clean_text(
                    item.get("pubDateTZ")
                )
            )

            articles.append(
                NewsArticle(
                    title=title,
                    description=description,
                    source=source,
                    url=url,
                    published_at=published_at,
                    category=category,
                )
            )

        # ----------------------------------------------------
        # Freshness.
        # ----------------------------------------------------

        articles = _filter_fresh_articles(
            articles
        )

        articles = _deduplicate(
            articles
        )

        if len(articles) >= limit:
            break

        next_page = payload.get(
            "nextPage"
        )

        if not next_page:
            break

    print(
        f"[ZOE NEWS] "
        f"NewsData returned "
        f"{len(articles)} fresh article(s)"
    )

    # --------------------------------------------------------
    # URL resolution.
    # --------------------------------------------------------

    articles = _resolve_article_urls(
        articles
    )

    # --------------------------------------------------------
    # Final safety passes.
    # --------------------------------------------------------

    articles = _filter_fresh_articles(
        articles
    )

    articles = _deduplicate(
        articles
    )

    articles = articles[:limit]

    _cache_set(
        cache_key,
        articles,
    )

    return articles


# ============================================================
# GOOGLE NEWS RSS
# ============================================================

def _google_news_url(
    query: str,
) -> str:

    encoded_query = urllib.parse.quote_plus(
        query
    )

    return (
        "https://news.google.com/rss/search"
        f"?q={encoded_query}"
        "&hl=en-US"
        "&gl=US"
        "&ceid=US:en"
    )


def _fetch_google_news(
    *,
    query: str,
    category: str,
    limit: int,
) -> list[NewsArticle]:

    limit = _normalize_limit(limit)

    # Search freshness words are unnecessary.
    search_query = _remove_freshness_words(
        query
    )

    if not search_query:
        search_query = query

    cache_key = (
        "google:"
        + category
        + ":"
        + search_query.lower()
    )

    cached = _cache_get(cache_key)

    if cached is not None:
        return cached[:limit]

    url = _google_news_url(
        search_query
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": BROWSER_USER_AGENT
        },
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=REQUEST_TIMEOUT,
        ) as response:

            data = response.read()

    except Exception as exc:

        print(
            "[ZOE NEWS] "
            "Google RSS request failed:",
            _redact_sensitive_text(exc),
        )

        return []

    try:

        root = ET.fromstring(data)

    except ET.ParseError as exc:

        print(
            "[ZOE NEWS] "
            "Google RSS parse failed:",
            repr(exc),
        )

        return []

    channel = root.find("channel")

    if channel is None:
        return []

    articles: list[NewsArticle] = []

    for item in channel.findall("item"):

        title = _clean_text(
            item.findtext("title")
        )

        description = _clean_text(
            item.findtext("description")
        )

        url = _clean_url(
            item.findtext("link")
        )

        published_at = _clean_text(
            item.findtext("pubDate")
        )

        if not title:
            continue

        # ----------------------------------------------------
        # Source.
        # ----------------------------------------------------

        source_element = item.find(
            "source"
        )

        source = ""

        if source_element is not None:

            source = _clean_text(
                source_element.text
            )

        # ----------------------------------------------------
        # Fallback:
        #
        # Headline - Publisher
        # ----------------------------------------------------

        if not source and " - " in title:

            possible_title, possible_source = (
                title.rsplit(
                    " - ",
                    1,
                )
            )

            if possible_source.strip():

                title = possible_title.strip()

                source = (
                    possible_source.strip()
                )

        if not source:
            source = "Unknown source"

        articles.append(
            NewsArticle(
                title=title,
                description=description,
                source=source,
                url=url,
                published_at=published_at,
                category=category,
            )
        )

    print(
        f"[ZOE NEWS] "
        f"Google RSS returned "
        f"{len(articles)} raw article(s)"
    )

    # --------------------------------------------------------
    # Freshness BEFORE URL resolution.
    # --------------------------------------------------------

    articles = _filter_fresh_articles(
        articles
    )

    articles = _deduplicate(
        articles
    )

    articles = articles[:limit]

    print(
        f"[ZOE NEWS] "
        f"Google RSS has "
        f"{len(articles)} fresh article(s)"
    )

    # --------------------------------------------------------
    # Resolve publisher URLs.
    #
    # Failure does NOT delete articles.
    # --------------------------------------------------------

    articles = _resolve_article_urls(
        articles
    )

    # --------------------------------------------------------
    # Final freshness.
    # --------------------------------------------------------

    articles = _filter_fresh_articles(
        articles
    )

    articles = _deduplicate(
        articles
    )

    articles = articles[:limit]

    _cache_set(
        cache_key,
        articles,
    )

    return articles


# ============================================================
# DEFAULT QUERIES
# ============================================================

FOOTBALL_DEFAULT_QUERY = (
    "football OR soccer"
)

F1_DEFAULT_QUERY = (
    '"Formula 1" OR F1 OR "F1 Grand Prix"'
)

WORLD_DEFAULT_QUERY = (
    "world"
)


# ============================================================
# INTERNAL FETCH
# ============================================================

def _get_news(
    *,
    category: str,
    query: str | None,
    limit: int,
) -> list[dict[str, Any]]:

    limit = _normalize_limit(limit)

    category = (
        category
        .lower()
        .strip()
    )

    if category not in {
        "f1",
        "football",
        "world",
    }:
        return []

    # --------------------------------------------------------
    # Explicit query.
    # --------------------------------------------------------

    if query and query.strip():

        search_query = _normalize_search_query(
            query
        )

    # --------------------------------------------------------
    # Category-only request.
    # --------------------------------------------------------

    elif category == "f1":

        search_query = F1_DEFAULT_QUERY

    elif category == "football":

        search_query = FOOTBALL_DEFAULT_QUERY

    else:

        search_query = WORLD_DEFAULT_QUERY

    print(
        "[ZOE NEWS] "
        f"Category={category} | "
        f"Query={search_query!r}"
    )

    # ========================================================
    # PRIMARY: NEWSDATA
    # ========================================================

    articles = _fetch_newsdata(
        query=search_query,
        category=category,
        limit=limit,
    )

    if articles:

        print(
            "[ZOE NEWS] "
            f"Using NewsData: "
            f"{len(articles)} article(s)"
        )

        return [
            article.to_dict()
            for article in articles[:limit]
        ]

    # ========================================================
    # FALLBACK: GOOGLE NEWS
    # ========================================================

    print(
        "[ZOE NEWS] "
        "NewsData returned no usable articles. "
        "Falling back to Google News."
    )

    articles = _fetch_google_news(
        query=search_query,
        category=category,
        limit=limit,
    )

    # ========================================================
    # FINAL SAFETY.
    # ========================================================

    articles = _filter_fresh_articles(
        articles
    )

    articles = _deduplicate(
        articles
    )

    articles = articles[:limit]

    print(
        "[ZOE NEWS] "
        f"Final result: "
        f"{len(articles)} article(s)"
    )

    return [
        article.to_dict()
        for article in articles
    ]


# ============================================================
# PUBLIC API
# ============================================================

def get_f1_news(
    query: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:

    return _get_news(
        category="f1",
        query=query,
        limit=limit,
    )


def get_football_news(
    query: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:

    return _get_news(
        category="football",
        query=query,
        limit=limit,
    )


def get_world_news(
    query: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:

    return _get_news(
        category="world",
        query=query,
        limit=limit,
    )


def search_news(
    query: str,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:

    query = _normalize_search_query(
        query
    )

    if not query:
        return []

    limit = _normalize_limit(limit)

    print(
        "[ZOE NEWS] "
        f"Search query={query!r}"
    )

    # --------------------------------------------------------
    # Explicit search is ALWAYS global.
    # --------------------------------------------------------

    articles = _fetch_newsdata(
        query=query,
        category="search",
        limit=limit,
    )

    if articles:

        return [
            article.to_dict()
            for article in articles[:limit]
        ]

    # --------------------------------------------------------
    # Google fallback.
    # --------------------------------------------------------

    articles = _fetch_google_news(
        query=query,
        category="search",
        limit=limit,
    )

    articles = _filter_fresh_articles(
        articles
    )

    articles = _deduplicate(
        articles
    )

    return [
        article.to_dict()
        for article in articles[:limit]
    ]


# ============================================================
# STATUS
# ============================================================

def news_service_status() -> dict[str, Any]:

    return {
        "newsdata_configured": bool(
            NEWSDATA_API_KEY
        ),
        "google_news_fallback": True,
        "google_news_url_resolution": True,
        "google_articles_kept_if_resolution_fails": True,
        "chromium_resolution": True,
        "concurrent_url_resolution": True,
        "google_resolve_timeout_seconds": (
            GOOGLE_RESOLVE_TIMEOUT
        ),
        "google_resolve_concurrency": (
            GOOGLE_RESOLVE_CONCURRENCY
        ),
        "news_max_age_hours": (
            NEWS_MAX_AGE_HOURS
        ),
        "future_article_tolerance_minutes": (
            FUTURE_ARTICLE_TOLERANCE_MINUTES
        ),
        "newsdata_page_size": (
            NEWSDATA_PAGE_SIZE
        ),
        "cache_ttl_seconds": CACHE_TTL,
        "url_resolution_cache_ttl_seconds": (
            URL_RESOLUTION_CACHE_TTL
        ),
        "default_limit": DEFAULT_LIMIT,
        "max_limit": MAX_LIMIT,
    }


# ============================================================
# CACHE CLEAR
# ============================================================

def clear_news_cache() -> None:
    _cache.clear()
    _url_resolution_cache.clear()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ZOE NEWS RETRIEVAL SERVICE")
    print("=" * 60)
    print()

    print(
        "Freshness window: "
        f"{NEWS_MAX_AGE_HOURS} hours"
    )

    print(
        "NewsData: primary"
    )

    print(
        "Google News RSS: fallback"
    )

    print(
        "Explicit queries: global search"
    )

    print(
        "Old articles: rejected"
    )

    print(
        "Undated articles: rejected"
    )

    print(
        "Google URL resolution failure: "
        "article is kept"
    )

    print()

    print("Examples:")

    print(
        "  latest world news"
    )

    print(
        "  latest Bangladesh news"
    )

    print(
        "  latest US news"
    )

    print(
        "  latest best AI model"
    )

    print(
        "  GPT-6 Astra"
    )

    print()

    print(
        "Type 'exit' or 'quit' to stop."
    )

    print()

    while True:

        try:

            query = input(
                "News > "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print()
            break

        if not query:
            continue

        if query.lower() in {
            "exit",
            "quit",
        }:
            break

        articles = search_news(
            query=query,
            limit=8,
        )

        print()

        if not articles:

            print(
                "No fresh articles found."
            )

            print()

            continue

        for index, article in enumerate(
            articles,
            start=1,
        ):

            print(
                f"{index}. "
                f"{article['title']}"
            )

            print(
                f"   Source: "
                f"{article['source']}"
            )

            print(
                f"   Published: "
                f"{article['published_at']}"
            )

            print(
                f"   URL: "
                f"{article['url']}"
            )

            if article["description"]:

                print(
                    f"   "
                    f"{article['description'][:300]}"
                )

            print()