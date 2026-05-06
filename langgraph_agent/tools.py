"""Playwright browser tools exposed to the LLM agent.

Each tool is a simple function that operates on a shared Playwright page.
No fallback logic — if a selector doesn't exist, Playwright raises an error.
"""

from __future__ import annotations

from typing import Optional

from playwright.async_api import Page

# Module-level page reference, set by the runner before the graph executes.
_page: Optional[Page] = None


def set_page(page: Page) -> None:
    """Set the shared Playwright page for all tools."""
    global _page
    _page = page


def get_page() -> Page:
    """Get the shared Playwright page. Raises if not set."""
    assert _page is not None, "Playwright page not initialized"
    return _page


async def navigate(url: str) -> str:
    """Navigate the browser to a URL. Returns the page title after load."""
    page = get_page()
    await page.goto(url, wait_until="domcontentloaded")
    title = await page.title()
    return f"Navigated to: {title} ({page.url})"


async def click(selector: str) -> str:
    """Click an element on the page identified by CSS selector."""
    page = get_page()
    await page.locator(selector).click()
    return f"Clicked '{selector}'"


async def fill_input(selector: str, text: str) -> str:
    """Fill a text input field identified by selector with the given text."""
    page = get_page()
    await page.locator(selector).fill(text)
    return f"Filled '{selector}' with: {text}"


async def press_key(selector: str, key: str) -> str:
    """Press a keyboard key on the element matching selector."""
    page = get_page()
    await page.locator(selector).press(key)
    return f"Pressed '{key}' on '{selector}'"


async def wait_for_selector(selector: str, timeout_ms: int = 10000) -> str:
    """Wait for an element matching selector to appear on the page."""
    page = get_page()
    await page.wait_for_selector(selector, timeout=timeout_ms)
    return f"Element '{selector}' appeared"


async def extract_links(selector: str) -> str:
    """Extract all link text and href from elements matching selector.

    Returns one link per line in format: title | url
    """
    page = get_page()
    links = await page.locator(selector).all()
    results = []
    for link in links:
        text = (await link.text_content() or "").strip()
        href = await link.get_attribute("href") or ""
        if text and href:
            results.append(f"{text} | {href}")
    return "\n".join(results) if results else "No links found"


async def get_page_text(selector: str) -> str:
    """Get text content of all elements matching a selector."""
    page = get_page()
    elements = await page.locator(selector).all_text_contents()
    return "\n".join(elements) if elements else "No text found"
