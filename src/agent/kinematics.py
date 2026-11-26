from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class TrajectorySample:
    t: float
    x: float
    y: float


def fitts_duration(
    start: Point2D,
    target: Point2D,
    target_width: float = 24.0,
    a_sec: float = 0.08,
    b_sec: float = 0.14,
    jitter_ratio: float = 0.12,
    rng: random.Random | None = None,
) -> float:
    """Compute movement duration T (seconds) via Fitts's Law with bounded jitter."""
    r = rng if rng is not None else random
    distance = math.hypot(target.x - start.x, target.y - start.y)
    if distance < 1e-6:
        return max(0.02, a_sec * (1.0 + r.uniform(-jitter_ratio, jitter_ratio)))

    width = max(1.0, float(target_width))
    id_bits = math.log2(1.0 + (distance / width))
    raw_t = a_sec + b_sec * id_bits
    jitter = r.uniform(-jitter_ratio, jitter_ratio)
    return max(0.04, raw_t * (1.0 + jitter))


def minimum_jerk_tau(tau: float) -> float:
    """Minimum-jerk position fraction s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5 for tau in [0, 1]."""
    u = max(0.0, min(1.0, float(tau)))
    return (10.0 * (u**3)) - (15.0 * (u**4)) + (6.0 * (u**5))


def cubic_bezier_point(
    u: float,
    p0: Point2D,
    p1: Point2D,
    p2: Point2D,
    p3: Point2D,
) -> Point2D:
    """Evaluate cubic Bezier curve at parameter u in [0, 1]."""
    t = max(0.0, min(1.0, float(u)))
    mt = 1.0 - t
    mt2 = mt * mt
    t2 = t * t

    x = (
        (mt2 * mt * p0.x)
        + (3.0 * mt2 * t * p1.x)
        + (3.0 * mt * t2 * p2.x)
        + (t2 * t * p3.x)
    )
    y = (
        (mt2 * mt * p0.y)
        + (3.0 * mt2 * t * p1.y)
        + (3.0 * mt * t2 * p2.y)
        + (t2 * t * p3.y)
    )
    return Point2D(x=x, y=y)


def ou_noise_series(
    steps: int,
    dt: float,
    theta: float = 8.0,
    mu: float = 0.0,
    sigma: float = 6.0,
    rng: random.Random | None = None,
) -> list[float]:
    """Generate mean-reverting Ornstein-Uhlenbeck noise values."""
    if steps <= 0:
        return []
    r = rng if rng is not None else random
    dt = max(1e-4, float(dt))
    sqrt_dt = math.sqrt(dt)

    val = mu
    series: list[float] = []
    for _ in range(steps):
        dw = r.gauss(0.0, 1.0) * sqrt_dt
        val = val + theta * (mu - val) * dt + sigma * dw
        series.append(val)
    return series


def build_cursor_trajectory(
    start: Point2D,
    target: Point2D,
    target_width: float = 24.0,
    dt: float = 0.012,
    curve_bend: float = 0.18,
    noise_sigma: float = 3.5,
    overshoot_prob: float = 0.15,
    rng: random.Random | None = None,
) -> list[TrajectorySample]:
    """Generate a smooth human-like cursor path from start to target."""
    r = rng if rng is not None else random
    step_dt = max(0.004, float(dt))

    dx = target.x - start.x
    dy = target.y - start.y
    dist = math.hypot(dx, dy)
    if dist < 1.0:
        return [
            TrajectorySample(t=0.0, x=start.x, y=start.y),
            TrajectorySample(t=step_dt, x=target.x, y=target.y),
        ]

    # Perpendicular unit vector for natural arc bend
    nx, ny = -dy / dist, dx / dist
    bend_scale = dist * curve_bend * r.uniform(-1.0, 1.0)
    p0 = start
    p3 = target
    p1 = Point2D(
        x=start.x + 0.33 * dx + nx * bend_scale,
        y=start.y + 0.33 * dy + ny * bend_scale,
    )
    p2 = Point2D(
        x=start.x + 0.67 * dx + nx * bend_scale * 0.7,
        y=start.y + 0.67 * dy + ny * bend_scale * 0.7,
    )

    duration = fitts_duration(start=start, target=target, target_width=target_width, rng=r)
    num_steps = max(4, int(math.ceil(duration / step_dt)))
    actual_dt = duration / num_steps

    noise_x = ou_noise_series(steps=num_steps, dt=actual_dt, sigma=noise_sigma, rng=r)
    noise_y = ou_noise_series(steps=num_steps, dt=actual_dt, sigma=noise_sigma, rng=r)

    samples: list[TrajectorySample] = []
    for i in range(num_steps + 1):
        t = i * actual_dt
        tau = i / num_steps
        # Blend linear progress with minimum-jerk profile so motion accelerates and settles smoothly
        s = minimum_jerk_tau(tau)
        # Taper noise to 0 at start and end so the pointer lands cleanly on target
        envelope = math.sin(math.pi * tau)
        pt = cubic_bezier_point(s, p0, p1, p2, p3)
        nx_val = noise_x[min(i, len(noise_x) - 1)] * envelope if 0 < i < num_steps else 0.0
        ny_val = noise_y[min(i, len(noise_y) - 1)] * envelope if 0 < i < num_steps else 0.0
        samples.append(TrajectorySample(t=round(t, 5), x=pt.x + nx_val, y=pt.y + ny_val))

    # Optional tiny overshoot + quick correction on longer/smaller-target moves
    if dist > 120.0 and r.random() < overshoot_prob:
        over_x = target.x + (dx / dist) * r.uniform(3.0, min(14.0, target_width * 0.5))
        over_y = target.y + (dy / dist) * r.uniform(3.0, min(14.0, target_width * 0.5))
        t_over = round(duration + 0.04, 5)
        t_settle = round(duration + 0.11, 5)
        samples.append(TrajectorySample(t=t_over, x=over_x, y=over_y))
        samples.append(TrajectorySample(t=t_settle, x=target.x, y=target.y))

    # Guarantee exact final landing coordinate
    samples[-1] = TrajectorySample(t=samples[-1].t, x=target.x, y=target.y)
    return samples


def scroll_impulse_steps(
    total_delta_y: float,
    bursts: int = 3,
    dt: float = 0.02,
    damping_lambda: float = 9.0,
    rng: random.Random | None = None,
) -> list[tuple[float, float]]:
    """Generate smooth damped scroll increments (delta_t, delta_y) summing to total_delta_y."""
    r = rng if rng is not None else random
    step_dt = max(0.005, float(dt))
    burst_count = max(1, int(bursts))

    # Split total scroll into uneven human bursts
    weights = [r.uniform(0.6, 1.4) for _ in range(burst_count)]
    w_sum = sum(weights)
    burst_amounts = [total_delta_y * (w / w_sum) for w in weights]

    steps_per_burst = max(6, int(0.28 / step_dt))
    decay = math.exp(-max(0.1, damping_lambda) * step_dt)

    raw_deltas: list[tuple[float, float]] = []
    current_t = 0.0
    velocity = 0.0

    for amt in burst_amounts:
        # Impulse scales with desired burst displacement and damping factor
        impulse = amt * (1.0 - decay) * 1.35
        for _ in range(steps_per_burst):
            velocity = velocity * decay + impulse
            impulse *= 0.55
            current_t = round(current_t + step_dt, 5)
            raw_deltas.append((current_t, velocity * step_dt))

    # Normalize so total scroll matches total_delta_y cleanly
    actual_total = sum(dy for _, dy in raw_deltas)
    if abs(actual_total) > 1e-6:
        scale = total_delta_y / actual_total
        return [(t, dy * scale) for t, dy in raw_deltas]
    return [(step_dt, float(total_delta_y))]