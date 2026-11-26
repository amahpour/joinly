import asyncio
import contextlib
import logging
import re
from typing import Any, ClassVar
from urllib.parse import parse_qs, urlparse

from playwright.async_api import Locator, Page

from joinly.providers.browser.platforms.base import BaseBrowserPlatformController
from joinly.settings import get_settings
from joinly.types import (
    MeetingChatHistory,
    MeetingChatMessage,
    MeetingParticipant,
    ProviderNotSupportedError,
)

logger = logging.getLogger(__name__)

_TIME_RX = re.compile(r"^\d{1,2}:\d{2}(?:\s*[AP]M)?$", re.IGNORECASE)
_MAX_MESSAGE_LENGTH = 1024


def _shown(locator: Locator) -> Locator:
    """The first of ``locator``'s elements that is visible, not the first in the page."""
    return locator.locator("visible=true").first


async def _appears(locator: Locator, timeout: float = 3000, state: str = "visible") -> bool:  # noqa: ASYNC109
    """Whether ``locator`` reaches ``state`` within ``timeout`` milliseconds."""
    try:
        await locator.wait_for(state=state, timeout=timeout)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return False
    return True


async def _press(button: Locator) -> None:
    """Press ``button`` with the pointer: Zoom opens its menus on a real press, not a script's.

    A button in a hidden bar is hovered first, which shows the bar; a script's click is
    the last resort.
    """
    if not await button.is_visible():
        box = await button.bounding_box()
        if box:
            await button.page.mouse.move(
                box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=4
            )
            await _appears(button, timeout=1500)
    if await button.is_visible():
        await button.click(timeout=1000)
        return
    try:
        await button.click(timeout=1000, force=True)
    except Exception:  # noqa: BLE001
        await button.evaluate("b => b.click()")


async def _log_state(button: Locator) -> None:
    """Log how ``button`` is shown and what overlays the page, to find a control Zoom moved."""
    with contextlib.suppress(Exception):
        state = await button.evaluate(
            """b => {
                const s = getComputedStyle(b), r = b.getBoundingClientRect();
                const dialogs = [...document.querySelectorAll('[role=dialog],[role=menu],[role=alertdialog]')]
                    .map(d => (d.getAttribute('aria-label') || d.innerText || '').trim().slice(0, 60));
                return {visibility: s.visibility, opacity: s.opacity, display: s.display,
                        rect: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)],
                        ariaHidden: !!b.closest('[aria-hidden="true"]'), inert: !!b.closest('[inert]'),
                        viewport: [innerWidth, innerHeight], dialogs: dialogs.slice(0, 5)};
            }"""
        )
        logger.warning("Zoom control state: %s", state)


def _mic_buttons(page: Page) -> tuple[Locator, Locator]:
    """The mute and unmute buttons, by their exact labels: one names the other's text."""
    return (
        page.locator('button[aria-label="mute my microphone" i]').first,
        page.locator('button[aria-label="unmute my microphone" i]').first,
    )


class ZoomBrowserPlatformController(BaseBrowserPlatformController):
    """Controller for managing Zoom browser meetings."""

    url_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?:https?://)?(?:[a-z0-9-]+\.)?zoom\.us/"
    )

    def __init__(self) -> None:
        """Initialize the Zoom browser platform controller."""
        self._state: dict[str, Any] = {}

    @property
    def active_speaker(self) -> str | None:
        """Get the name of the active speaker in the Zoom meeting."""
        return self._state.get("active_speaker")

    async def join(  # noqa: C901, PLR0912
        self,
        page: Page,
        url: str,
        name: str,
        passcode: str | None = None,
    ) -> None:
        """Join the Zoom meeting.

        Args:
            page: The Playwright page instance.
            url: The URL of the Zoom meeting.
            name: The name of the participant.
            passcode: The passcode for the meeting (if required).
        """
        # convert standard join URL to the web client format
        if re.search(r"/j/\d+", url):
            url = re.sub(r"/j/(\d+)", r"/wc/join/\1", url)
            logger.debug("Rewrote Zoom join URL to web client format: %s", url)

        await page.goto(url, wait_until="load", timeout=20000)
        if await page.get_by_text("invalid").is_visible():
            msg = "Meeting link is invalid."
            raise ValueError(msg)

        async def _click_task(page: Page, text: str) -> None:
            await page.locator("button", has_text=text).first.click(timeout=0)

        pre_tasks = [
            asyncio.create_task(_click_task(page, t))
            for t in ["accept cookies", "i agree"]
        ]

        try:
            name_field = page.locator("#input-for-name, #inputname")
            await name_field.fill(name, timeout=20000)

            passcode_field = page.locator("input[type='password']")
            if await passcode_field.is_visible():
                if passcode is not None:
                    await passcode_field.fill(passcode, timeout=1000)
                else:
                    msg = "Passcode is required but not provided."
                    raise ValueError(msg)

            join_btn = page.locator("button", has_text="join").first
            await join_btn.click(timeout=1000)

            await page.wait_for_timeout(2000)
            if await join_btn.is_visible():
                with contextlib.suppress(Exception):
                    await join_btn.click(timeout=1000)

            await page.wait_for_timeout(1000)
            if await page.get_by_text("incorrect password").is_visible():
                pwd_param = parse_qs(urlparse(url).query).get("pwd", [None])[0]
                if passcode := pwd_param.split(".1")[-1] if pwd_param else None:
                    await passcode_field.fill(passcode, timeout=1000)
                    await join_btn.click(timeout=1000)
                else:
                    msg = "Incorrect passcode provided."
                    raise ValueError(msg)
        finally:
            for task in pre_tasks:
                if not task.done():
                    task.cancel()

        if not await self._check_joined(page):
            msg = "Join check failed: Failed to join the Zoom meeting."
            raise RuntimeError(msg)

        await self._setup_active_speaker_observer(page)

    async def leave(self, page: Page) -> None:
        """Leave the Zoom meeting using the icon-based button.

        Zoom's web client hides its control bar, from sight and from the accessibility
        tree, until the pointer moves. So the button is found by its label, not its role,
        and pressed directly when the bar is still hidden: a role lookup read it as
        missing, and the bot stayed in the call.
        """
        await self._activate_controls(page)

        leave_btn = page.locator('button[aria-label="leave" i]').first
        if not await _appears(leave_btn, state="attached"):
            await self._log_buttons(page)
            msg = "Leave button not found or not visible."
            raise RuntimeError(msg)
        await _log_state(leave_btn)
        await _press(leave_btn)

        leave_btn_confirm = _shown(page.locator("button", has_text="leave meeting"))
        if not await _appears(leave_btn_confirm):
            with contextlib.suppress(Exception):
                await self._activate_controls(page)
                await _press(leave_btn)

        if not await _appears(leave_btn_confirm):
            await self._log_buttons(page)
            await _log_state(leave_btn)
            msg = "Leave meeting confirmation button not found or not visible."
            raise RuntimeError(msg)
        await leave_btn_confirm.click(timeout=1000)
        await page.wait_for_timeout(500)

    async def send_chat_message(self, page: Page, message: str) -> None:
        """Send a chat message in Zoom."""
        if len(message) > _MAX_MESSAGE_LENGTH:
            msg = (
                f"Message exceeds the maximum length of {_MAX_MESSAGE_LENGTH} "
                f"characters, got {len(message)}."
            )
            raise ValueError(msg)

        await self._open_chat(page)

        chat_input = page.locator("div[contenteditable='true']")
        if not await chat_input.is_visible():
            msg = "Chat input not found or not visible."
            raise RuntimeError(msg)
        await chat_input.click(timeout=1000)
        await page.wait_for_timeout(200)
        await chat_input.fill(message)
        await page.wait_for_timeout(500)
        await page.keyboard.press("Enter")

    async def get_chat_history(self, page: Page) -> MeetingChatHistory:
        """Return a Zoom in-meeting chat history.

        Args:
            page: The Playwright page instance.

        Returns:
            MeetingChatHistory: The chat history of the meeting.
        """
        await self._open_chat(page)

        messages: list[MeetingChatMessage] = []

        panel = page.locator('div[role="application"][aria-label="Chat Message List"]')
        rows = await panel.locator('[role="row"][aria-label]').all()

        for row in rows:
            aria = await row.get_attribute("aria-label") or ""
            parts = [p.strip() for p in aria.split(",")]

            sender: str | None = None
            ts: str | None = None

            if parts and len(parts) >= 3:  # noqa: PLR2004
                first = parts[0]
                sender = (
                    (first.split(" to ")[0].strip() or None)
                    if " to " in first
                    else (first or None)
                )

                raw_time = re.sub(r"[\u00A0\u202F]", "", parts[1]).strip()
                if _TIME_RX.fullmatch(raw_time):
                    ts = raw_time

                text = ",".join(parts[2:]).strip()

                if text:
                    messages.append(
                        MeetingChatMessage(text=text, timestamp=ts, sender=sender)
                    )

        return MeetingChatHistory(messages=messages)

    async def get_participants(self, page: Page) -> list[MeetingParticipant]:
        """Get the list of participants in the Zoom meeting.

        Args:
            page: The Playwright page instance.

        Returns:
            list[MeetingParticipant]: A list of participants in the meeting.
        """
        participants_list = page.locator(
            'div[role="list"][aria-label^="participants" i]'
        )
        is_participant_list_visible = await participants_list.is_visible()

        participants_button = page.get_by_role(
            "button", name=re.compile(r"participants list", re.IGNORECASE)
        )
        if not is_participant_list_visible:
            await self._activate_controls(page)
            if not await participants_button.is_visible():
                msg = "Participants button not found or not visible."
                raise RuntimeError(msg)
            await participants_button.click(timeout=1000)
            if not await participants_list.is_visible():
                with contextlib.suppress(Exception):
                    await participants_button.click(timeout=1000)
                await page.wait_for_timeout(1000)

        participants: list[MeetingParticipant] = []
        for item in await participants_list.locator("div.participants-li").all():
            if aria_label := await item.get_attribute("aria-label"):
                labels = aria_label.split(",")
                name = labels[0].strip()
                infos = labels[1:] if len(labels) > 1 else []
                participants.append(MeetingParticipant(name=name, infos=infos))

        await participants_button.click(timeout=1000)

        return participants

    async def mute(self, page: Page) -> None:
        """Mute the microphone in Zoom."""
        await self._activate_controls(page)

        mute_btn, unmute_btn = _mic_buttons(page)
        if not await _appears(mute_btn.or_(unmute_btn).first, state="attached"):
            await self._log_buttons(page)
            msg = "Mute button not found or not visible."
            raise RuntimeError(msg)
        if await mute_btn.count():
            await _press(mute_btn)

    async def unmute(self, page: Page) -> None:
        """Unmute the microphone in Zoom."""
        await self._activate_controls(page)

        mute_btn, unmute_btn = _mic_buttons(page)
        if not await _appears(unmute_btn.or_(mute_btn).first, state="attached"):
            await self._log_buttons(page)
            msg = "Unmute button not found or not visible."
            raise RuntimeError(msg)
        if await unmute_btn.count():
            await _press(unmute_btn)
            if await _appears(unmute_btn, timeout=500, state="attached") and not await mute_btn.count():
                with contextlib.suppress(Exception):
                    await _press(unmute_btn)

    async def share_screen(self, page: Page) -> None:
        """Start sharing screen in the Zoom meeting.

        Args:
            page: The Playwright page instance.
        """
        # Zoom screen sharing is not yet working reliably.
        msg = "Screen sharing is not yet supported for Zoom."
        raise ProviderNotSupportedError(msg)
        # --- unreachable: kept for when Zoom support is fixed ---
        await self._activate_controls(page)

        share_btn = page.get_by_role(
            "button", name=re.compile(r"share screen", re.IGNORECASE)
        )
        if not await share_btn.is_visible():
            msg = "Share Screen button not found or not visible."
            raise RuntimeError(msg)
        await share_btn.click(timeout=2000)
        await page.wait_for_timeout(1000)

    async def stop_sharing(self, page: Page) -> None:
        """Stop sharing screen in the Zoom meeting.

        Args:
            page: The Playwright page instance.
        """
        msg = "Screen sharing is not yet supported for Zoom."
        raise ProviderNotSupportedError(msg)
        # --- unreachable: kept for when Zoom support is fixed ---
        await self._activate_controls(page)

        stop_btn = page.get_by_role(
            "button", name=re.compile(r"stop shar", re.IGNORECASE)
        )
        if not await stop_btn.is_visible():
            msg = "Stop Share button not found or not visible."
            raise RuntimeError(msg)
        await stop_btn.click(timeout=2000)
        await page.wait_for_timeout(500)

    async def _check_joined(self, page: Page, timeout: float = 10) -> bool:  # noqa: ASYNC109
        """Check if the Zoom meeting has been joined successfully.

        Args:
            page: The Playwright page instance.
            timeout: The timeout in seconds for checking the join status.

        Returns:
            bool: True if joined, False otherwise.
        """
        locators = [
            page.locator("span >> text=/joining/i"),
            page.locator("span >> text=/we've let them know you're here/i"),
            page.get_by_role("button", name=re.compile(r"leave", re.IGNORECASE)),
        ]

        tasks = [
            asyncio.create_task(loc.wait_for(state="visible", timeout=0))
            for loc in locators
        ]

        try:
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED, timeout=timeout
            )
            return any(not task.exception() for task in done)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def _activate_controls(self, page: Page) -> None:
        """Activate control bar: a click, then the pointer over the bar's edge, which keeps it shown.

        A notice Zoom keeps open until it is dismissed ("Got it") hides the rest of the
        page from the accessibility tree and keeps the bar's menus from opening, so it is
        dismissed first.
        """
        notices = page.locator("button", has_text=re.compile(r"^\s*got it\s*$", re.IGNORECASE))
        for _ in range(3):
            notice = _shown(notices)
            if not await notice.count():
                break
            with contextlib.suppress(Exception):
                await notice.click(timeout=1000)
                await page.wait_for_timeout(200)
        await page.mouse.click(640, 360)
        await page.mouse.move(640, 600, steps=4)
        await page.wait_for_timeout(100)

    async def _log_buttons(self, page: Page) -> None:
        """Log the names of the buttons shown, so a control Zoom renamed can be found."""
        with contextlib.suppress(Exception):
            names = await page.locator("button").evaluate_all(
                "els => els.filter(e => e.offsetParent !== null)"
                ".map(e => (e.getAttribute('aria-label') || e.innerText || '').trim().slice(0, 40))"
                ".filter(Boolean).slice(0, 40)"
            )
            logger.warning("Zoom buttons shown: %s", names)

    async def _open_chat(self, page: Page) -> None:
        """Open the chat in the Zoom meeting."""
        chat_input = page.locator("div[contenteditable='true']")
        is_chat_visible = await chat_input.is_visible()

        if not is_chat_visible:
            await self._activate_controls(page)
            chat_button = page.get_by_role(
                "button", name=re.compile(r"chat panel", re.IGNORECASE)
            )
            if not await chat_button.is_visible():
                msg = "Chat button not found or not visible."
                raise RuntimeError(msg)
            await chat_button.click(timeout=1000)
            if not await chat_input.is_visible():
                with contextlib.suppress(Exception):
                    await chat_button.click(timeout=1000)
                await page.wait_for_timeout(1000)

    async def _setup_active_speaker_observer(self, page: Page) -> None:
        """Setup the active speaker observer for Zoom."""
        await page.expose_binding(
            "report",
            lambda _, name: self._state.update({"active_speaker": name}),
        )
        await page.evaluate(
            """
            (nameArg) => {
                const emit = n => window.report(n);
                const find = () => {
                    const selectors = [
                        'div.single-main-container__video-frame span',
                        'div.speaker-active-container__video-frame span',
                        'div.speaker-bar-container__video-frame--active span',
                        'div.speaker-bar-container__video-frame span',
                    ];

                    const name = selectors
                        .flatMap(sel => Array.from(document.querySelectorAll(sel)))
                        .map(el => el?.textContent?.trim())
                        .find(text => text && text.length > 0 && text !== nameArg);
                    return name || null;
                };

                let last = null, cur;
                new MutationObserver(() => {
                    cur = find();
                    if (cur !== last) { last = cur; emit(cur); }
                }).observe(
                    document,
                    {
                        subtree: true,
                        childList: true,
                        attributes: true,
                        attributeFilter: ['class']
                    }
                );
                emit(find());
            }
            """,
            get_settings().name,
        )
