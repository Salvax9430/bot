from __future__ import annotations

import asyncio
from typing import Any

try:
    import pyautogui  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    pyautogui = None

from agent.kinematics import Point2D, build_cursor_trajectory, scroll_impulse_steps
from agent.perception import PerceptionEngine, ScreenRegion
from agent.typing_model import TypingModel


def _require_pyautogui() -> Any:
    if pyautogui is None:
        raise RuntimeError(
            "Optional dependency 'pyautogui' is not installed. Install it with: pip install pyautogui"
        )
    return pyautogui


class GuiInputBridge:
    """Executes human-paced mouse, scroll, and typing actions with a stop/kill switch."""

    def __init__(
        self,
        typing_model: TypingModel | None = None,
        perception: PerceptionEngine | None = None,
        stop_event: asyncio.Event | None = None,
        fail_safe: bool = True,
    ) -> None:
        self.typing_model = typing_model if typing_model is not None else TypingModel()
        self.perception = perception
        self.stop_event = stop_event if stop_event is not None else asyncio.Event()
        if pyautogui is not None:
            pyautogui.FAILSAFE = fail_safe
            # Keep pyautogui's built-in pause near zero because we pace every step explicitly
            pyautogui.PAUSE = 0.0

    def trigger_stop(self) -> None:
        self.stop_event.set()

    def clear_stop(self) -> None:
        self.stop_event.clear()

    def _is_stopped(self) -> bool:
        return self.stop_event.is_set()

    def current_pointer(self) -> Point2D:
        pag = _require_pyautogui()
        pos = pag.position()
        return Point2D(x=float(pos.x), y=float(pos.y))

    async def move_to(
        self,
        x: float,
        y: float,
        target_width: float = 24.0,
        hover_pause_sec: float = 0.06,
    ) -> dict[str, Any]:
        """Move cursor along a minimum-jerk + Bezier + OU-noise trajectory."""
        if self._is_stopped():
            return {"ok": False, "stopped": True, "error": "GUI execution stopped."}

        pag = _require_pyautogui()
        start = self.current_pointer()
        target = Point2D(x=float(x), y=float(y))
        samples = build_cursor_trajectory(start=start, target=target, target_width=target_width)

        prev_t = 0.0
        for sample in samples:
            if self._is_stopped():
                return {"ok": False, "stopped": True, "error": "Stopped mid-trajectory."}
            dt = max(0.0, sample.t - prev_t)
            if dt > 0:
                await asyncio.sleep(dt)
            prev_t = sample.t
            pag.moveTo(int(round(sample.x)), int(round(sample.y)))

        if hover_pause_sec > 0 and not self._is_stopped():
            await asyncio.sleep(hover_pause_sec)

        final_pos = self.current_pointer()
        return {
            "ok": True,
            "start": {"x": start.x, "y": start.y},
            "end": {"x": final_pos.x, "y": final_pos.y},
            "samples": len(samples),
        }

    async def click_at(
        self,
        x: float,
        y: float,
        button: str = "left",
        clicks: int = 1,
        target_width: float = 24.0,
        verify_region: ScreenRegion | None = None,
    ) -> dict[str, Any]:
        """Move smoothly to (x, y), click, and optionally check SSIM visual change."""
        if self._is_stopped():
            return {"ok": False, "stopped": True, "error": "GUI execution stopped."}

        pag = _require_pyautogui()
        before_img = (
            self.perception.capture(verify_region)
            if (self.perception is not None and verify_region is not None)
            else None
        )

        move_res = await self.move_to(x=x, y=y, target_width=target_width)
        if not move_res.get("ok"):
            return move_res

        if self._is_stopped():
            return {"ok": False, "stopped": True, "error": "Stopped before click."}

        pag.click(x=int(round(x)), y=int(round(y)), clicks=max(1, int(clicks)), button=button)
        await asyncio.sleep(0.08)

        result: dict[str, Any] = {
            "ok": True,
            "x": int(round(x)),
            "y": int(round(y)),
            "button": button,
            "clicks": clicks,
        }

        if before_img is not None and self.perception is not None:
            after_img = self.perception.capture(verify_region)
            result["visual_diff"] = self.perception.compare_frames(before_img, after_img)

        return result

    async def scroll_human(
        self,
        total_delta_y: float,
        bursts: int = 3,
    ) -> dict[str, Any]:
        """Scroll using momentum-damped bursts rather than a single jump."""
        if self._is_stopped():
            return {"ok": False, "stopped": True, "error": "GUI execution stopped."}

        pag = _require_pyautogui()
        steps = scroll_impulse_steps(total_delta_y=total_delta_y, bursts=bursts)
        prev_t = 0.0
        applied_total = 0.0

        for t_stamp, dy in steps:
            if self._is_stopped():
                return {
                    "ok": False,
                    "stopped": True,
                    "scrolled_so_far": round(applied_total, 2),
                    "error": "Stopped mid-scroll.",
                }
            dt = max(0.0, t_stamp - prev_t)
            if dt > 0:
                await asyncio.sleep(dt)
            prev_t = t_stamp
            # pyautogui.scroll expects integer wheel units; accumulate fractional residuals cleanly
            applied_total += dy
            wheel_step = int(round(dy / 40.0))
            if wheel_step != 0:
                pag.scroll(wheel_step)

        return {
            "ok": True,
            "target_delta_y": total_delta_y,
            "scrolled_approx": round(applied_total, 2),
        }

    async def type_human(
        self,
        text: str,
        allow_typos: bool = True,
    ) -> dict[str, Any]:
        """Type text using log-normal keystroke timing and optional typo + backspace."""
        if self._is_stopped():
            return {"ok": False, "stopped": True, "error": "GUI execution stopped."}

        pag = _require_pyautogui()
        events = self.typing_model.plan_text(text=text, allow_typos=allow_typos)
        typed_chars = 0
        corrections = 0

        for ev in events:
            if self._is_stopped():
                return {
                    "ok": False,
                    "stopped": True,
                    "typed_chars": typed_chars,
                    "corrections": corrections,
                    "error": "Stopped mid-typing.",
                }
            if ev.delay_sec > 0:
                await asyncio.sleep(ev.delay_sec)

            if ev.kind == "char":
                if ev.value == "\n":
                    pag.press("enter")
                else:
                    pag.write(ev.value, interval=0.0)
                typed_chars += 1
            elif ev.kind == "backspace":
                pag.press("backspace")
                corrections += 1
            elif ev.kind == "pause":
                continue

        return {
            "ok": True,
            "typed_chars": typed_chars,
            "corrections": corrections,
            "total_events": len(events),
        }

    def as_tool_map(self) -> dict[str, Any]:
        """Return callable tool functions ready to wire into AgentLoop.extra_tools."""

        async def _gui_move(args: dict[str, Any]) -> dict[str, Any]:
            return await self.move_to(
                x=float(args.get("x", 0.0)),
                y=float(args.get("y", 0.0)),
                target_width=float(args.get("target_width", 24.0)),
                hover_pause_sec=float(args.get("hover_pause_sec", 0.06)),
            )

        async def _gui_click(args: dict[str, Any]) -> dict[str, Any]:
            region_raw = args.get("verify_region")
            region = (
                ScreenRegion(
                    left=int(region_raw["left"]),
                    top=int(region_raw["top"]),
                    width=int(region_raw["width"]),
                    height=int(region_raw["height"]),
                )
                if isinstance(region_raw, dict)
                else None
            )
            return await self.click_at(
                x=float(args.get("x", 0.0)),
                y=float(args.get("y", 0.0)),
                button=str(args.get("button", "left")),
                clicks=int(args.get("clicks", 1)),
                target_width=float(args.get("target_width", 24.0)),
                verify_region=region,
            )

        async def _gui_scroll(args: dict[str, Any]) -> dict[str, Any]:
            return await self.scroll_human(
                total_delta_y=float(args.get("total_delta_y", -300.0)),
                bursts=int(args.get("bursts", 3)),
            )

        async def _gui_type(args: dict[str, Any]) -> dict[str, Any]:
            return await self.type_human(
                text=str(args.get("text", "")),
                allow_typos=bool(args.get("allow_typos", True)),
            )

        return {
            "gui_move": _gui_move,
            "gui_click": _gui_click,
            "gui_scroll": _gui_scroll,
            "gui_type": _gui_type,
        }
