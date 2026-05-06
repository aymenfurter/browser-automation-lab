"""Playwright browser tools registered with the Copilot SDK.

Each tool uses @define_tool with a Pydantic model for parameters.
The Copilot runtime calls these tools autonomously as part of its planning loop.
No fallback logic — if a selector doesn't exist, Playwright raises an error.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from copilot import define_tool
from playwright.async_api import Page

# Module-level page reference, set by the runner before the session starts.
_page: Page | None = None


def set_page(page: Page) -> None:
    """Set the shared Playwright page for all tools."""
    global _page
    _page = page


def get_page() -> Page:
    """Get the shared Playwright page. Raises if not set."""
    assert _page is not None, "Playwright page not initialized"
    return _page


# ─── Tool parameter models ──────────────────────────────────────────────────


class NavigateParams(BaseModel):
    url: str = Field(description="URL to navigate to")


class FillInputParams(BaseModel):
    selector: str = Field(description="CSS selector of the input field")
    text: str = Field(description="Text to type into the field")


class PressKeyParams(BaseModel):
    selector: str = Field(description="CSS selector of the element")
    key: str = Field(description="Key to press (e.g. 'Enter', 'Tab')")


class WaitForSelectorParams(BaseModel):
    selector: str = Field(description="CSS selector to wait for")
    timeout_ms: int = Field(default=10000, description="Timeout in milliseconds")


class ExtractLinksParams(BaseModel):
    selector: str = Field(description="CSS selector for <a> elements to extract")


class GetPageTextParams(BaseModel):
    selector: str = Field(description="CSS selector of elements to get text from")


# ─── Tool definitions ───────────────────────────────────────────────────────


@define_tool(description="Navigate the browser to a URL. Returns the page title after load.")
async def navigate_browser(params: NavigateParams) -> str:
    page = get_page()
    await page.goto(params.url, wait_until="domcontentloaded")
    title = await page.title()
    return f"Navigated to: {title} ({page.url})"


@define_tool(description="Fill a text input field identified by CSS selector with the given text.")
async def fill_input(params: FillInputParams) -> str:
    page = get_page()
    await page.locator(params.selector).fill(params.text)
    return f"Filled '{params.selector}' with: {params.text}"


@define_tool(description="Press a keyboard key on the element matching CSS selector.")
async def press_key(params: PressKeyParams) -> str:
    page = get_page()
    await page.locator(params.selector).press(params.key)
    return f"Pressed '{params.key}' on '{params.selector}'"


@define_tool(description="Wait for an element matching CSS selector to appear on the page.")
async def wait_for_selector(params: WaitForSelectorParams) -> str:
    page = get_page()
    await page.wait_for_selector(params.selector, timeout=params.timeout_ms)
    return f"Element '{params.selector}' appeared"


@define_tool(
    description=(
        "Extract all link text and href from <a> elements matching CSS selector. "
        "Returns one link per line in format: title | url"
    )
)
async def extract_links(params: ExtractLinksParams) -> str:
    page = get_page()
    links = await page.locator(params.selector).all()
    results = []
    for link in links:
        text = (await link.text_content() or "").strip()
        href = await link.get_attribute("href") or ""
        if text and href:
            results.append(f"{text} | {href}")
    return "\n".join(results) if results else "No links found"


@define_tool(description="Get the text content of all elements matching a CSS selector.")
async def get_page_text(params: GetPageTextParams) -> str:
    page = get_page()
    elements = await page.locator(params.selector).all_text_contents()
    return "\n".join(elements) if elements else "No text found"


ALL_TOOLS = [
    navigate_browser,
    fill_input,
    press_key,
    wait_for_selector,
    extract_links,
    get_page_text,
]
