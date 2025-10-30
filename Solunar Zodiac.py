# Heavily cleaned and reorganized version of the original script.
"""Solunar/Zodiac-style fishing window calculator.

Configuration (edit these values for your location):
- Latitude: decimal degrees (positive north)
- Longitude: decimal degrees (positive east)
- timezone: numeric offset from UTC in hours (e.g. -4 for EDT)
- day_offset: integer number of days to shift the report (0 for today)

This script is refactored so it can be imported without side-effects. Run
as a script to print the suggested fishing windows for the configured date.
"""
# ----------------------- User configuration -----------------------
Latitude = 40.0491
Longitude = -75.026
timezone = -4  # numeric offset from UTC in hours
day_offset = 0

# define major/minor window sizes (hours)
MAJOR_HOURS = 1.25   # around moon transit/underfoot
MINOR_HOURS = 0.6    # around moonrise/moonset

import argparse
import math
import time
import datetime
from typing import Optional, Tuple, List, Dict


# Derived timezone objects (simple fixed-offset timezone based on numeric offset)
LOCAL_TZ = datetime.timezone(datetime.timedelta(hours=int(timezone)))
TZ_NAME = f"UTC{timezone:+d}"


def hrmn(frac: float) -> str:
    """Format a fractional-day value (0..1) as 12-hour HH:MMAM/PM string.

    Uses integer-minute rounding to produce stable results.
    """
    frac = float(frac) % 1.0
    total_minutes = int(round(frac * 24 * 60)) % (24 * 60)
    hour = total_minutes // 60
    minute = total_minutes % 60
    ampm = "AM" if hour < 12 else "PM"
    hour12 = hour % 12 or 12
    return f"{hour12:02d}:{minute:02d}{ampm}"


def sunlight(lat: float, lon: float, timesec: float) -> List[float]:
    """Return approximate solar metrics: [duration, sunrise_frac, noon_frac, sunset_frac].

    All fractional-day values are relative to local midnight (the function uses
    a numeric `timezone` value for the noon calculation). This is a compact
    port of the original algorithm with clearer variable names.
    """
    rad = math.radians
    deg = math.degrees
    sin = math.sin
    cos = math.cos
    tan = math.tan

    # tiny offset to avoid JD edge cases
    E2 = 0.1 / 24
    JD = (timesec / 86400.0) + 2440587.5 + E2
    JC = (JD - 2451545.0) / 36525

    gmls = 280.46646 + JC * (36000.76983 + JC * 0.0003032)
    gmls %= 360
    gmas = 357.52911 + JC * (35999.05029 - 0.0001537 * JC)

    sec = (
        sin(rad(gmas)) * (1.914602 - JC * (0.004817 + 0.000014 * JC))
        + sin(rad(2 * gmas)) * (0.019993 - 0.000101 * JC)
        + sin(rad(3 * gmas)) * 0.000289
    )

    stl = gmls + sec
    sal = stl - 0.00569 - 0.00478 * sin(rad(125.04 - 1934.136 * JC))

    moe = 23 + (26 + ((21.448 - JC * (46.815 + JC * (0.00059 - JC * 0.001813)))) / 60) / 60
    oc = moe + 0.00256 * cos(rad(125.04 - 1934.136 * JC))

    sd = deg(math.asin(sin(rad(oc)) * sin(rad(sal))))
    # solar zenith angle correction for sunrise/sunset
    sunrised = deg(math.acos(cos(rad(90.833)) / (cos(rad(lat)) * cos(rad(sd))) - tan(rad(lat)) * tan(rad(sd))))

    eeo = round(0.016708634 - JC * (0.000042037 + 0.0000001267 * JC), 2)
    vary = round(tan(rad(oc / 2)) * tan(rad(oc / 2)), 2)

    et = 4 * deg(
        vary * sin(2 * rad(gmls))
        - 2 * eeo * sin(rad(gmas))
        + 4 * eeo * vary * sin(rad(gmas)) * cos(2 * rad(gmls))
        - 0.5 * vary * vary * sin(4 * rad(gmls))
        - 1.25 * eeo * eeo * sin(2 * rad(gmas))
    )

    noon = (720 - 4 * lon - et + timezone * 60) / 1440
    sunrise = noon - sunrised * 4 / 1440
    sunset = noon + sunrised * 4 / 1440
    duration = 8 * sunrised
    return [duration, sunrise, noon, sunset]


def frac_day_to_local_dt(frac: float, ref_date: datetime.date, tz: Optional[datetime.tzinfo] = None) -> datetime.datetime:
    """Convert fractional-day (0..1) to a timezone-aware local datetime on ref_date."""
    sec = (frac % 1.0) * 24 * 3600
    sec = sec % (24 * 3600)
    hh = int(sec // 3600)
    mm = int((sec % 3600) // 60)
    ss = int(sec % 60)
    if tz is None:
        tz = LOCAL_TZ
    return datetime.datetime(ref_date.year, ref_date.month, ref_date.day, hh, mm, ss, tzinfo=tz)


def to_local(dt_utc: Optional[datetime.datetime]) -> Optional[datetime.datetime]:
    """Convert a UTC datetime (naive or aware) to the configured local timezone."""
    if dt_utc is None:
        return None
    if dt_utc.tzinfo is None:
        dt = dt_utc.replace(tzinfo=datetime.timezone.utc)
    else:
        dt = dt_utc
    return dt.astimezone(LOCAL_TZ)


def fmt(dt: Optional[datetime.datetime]) -> str:
    if dt is None:
        return "--:--"
    return dt.astimezone(LOCAL_TZ).strftime('%m/%d/%Y %I:%M%p')


def make_window(center_dt: datetime.datetime, hours_before: float, hours_after: float) -> Tuple[datetime.datetime, datetime.datetime]:
    return (center_dt - datetime.timedelta(hours=hours_before), center_dt + datetime.timedelta(hours=hours_after))


def merge_windows(windows: List[Tuple[datetime.datetime, datetime.datetime]]) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    if not windows:
        return []
    windows = sorted(windows, key=lambda w: w[0])
    merged: List[Tuple[datetime.datetime, datetime.datetime]] = [windows[0]]
    for start, end in windows[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


# ----------------- Lunar math & event finding (low precision) -----------------


def jd_from_datetime(dt: datetime.datetime) -> float:
    if dt.tzinfo is not None:
        dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    year = dt.year
    month = dt.month
    day = dt.day + (dt.hour + (dt.minute + dt.second / 60.0) / 60.0) / 24.0
    if month <= 2:
        year -= 1
        month += 12
    A = year // 100
    B = 2 - A + A // 4
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
    return jd


def gmst_from_jd(jd: float) -> float:
    T = (jd - 2451545.0) / 36525.0
    gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T * T - T * T * T / 38710000.0
    return gmst % 360.0


def lst_deg(jd: float, lon_deg: float) -> float:
    return (gmst_from_jd(jd) + lon_deg) % 360.0


def obliquity_of_ecliptic(T: float) -> float:
    eps0 = 23.4392911111111 - 0.0130041666667 * T - 1.6666667e-7 * T * T + 5.0277778e-7 * T * T * T
    return eps0


def moon_position(dt: datetime.datetime) -> Dict[str, float]:
    jd = jd_from_datetime(dt)
    T = (jd - 2451545.0) / 36525.0
    Lp = (218.3164477 + 481267.88123421 * T - 0.0015786 * T * T + T**3 / 538841.0 - T**4 / 65194000.0) % 360.0
    D = (297.8501921 + 445267.1114034 * T - 0.0018819 * T * T + T**3 / 545868.0 - T**4 / 113065000.0) % 360.0
    M = (357.5291092 + 35999.0502909 * T - 0.0001536 * T * T + T**3 / 24490000.0) % 360.0
    Mp = (134.9633964 + 477198.8675055 * T + 0.0087414 * T * T + T**3 / 69699.0 - T**4 / 14712000.0) % 360.0
    F = (93.2720950 + 483202.0175233 * T - 0.0036539 * T * T - T**3 / 3526000.0 + T**4 / 863310000.0) % 360.0
    r = math.radians
    lon = Lp + 6.289 * math.sin(r(Mp)) + 1.274 * math.sin(r(2 * D - Mp)) + 0.658 * math.sin(r(2 * D))
    lon += 0.214 * math.sin(r(2 * Mp)) - 0.186 * math.sin(r(M)) - 0.059 * math.sin(r(2 * D - 2 * Mp))
    lon += -0.057 * math.sin(r(2 * D - Mp - M)) + 0.053 * math.sin(r(2 * D + Mp)) + 0.046 * math.sin(r(2 * D - M))
    lon += 0.041 * math.sin(r(Mp - M)) - 0.035 * math.sin(r(D)) - 0.031 * math.sin(r(Mp + M))
    lon = lon % 360.0
    lat = 5.128 * math.sin(r(F)) + 0.280 * math.sin(r(Mp + F)) + 0.277 * math.sin(r(Mp - F))
    lat += 0.173 * math.sin(r(2 * D - F)) + 0.055 * math.sin(r(2 * D + F - Mp)) + 0.046 * math.sin(r(2 * D - F - Mp))
    distance = 385000.56 - 20905.355 * math.cos(r(Mp)) - 3699.111 * math.cos(r(2 * D - Mp))
    distance += -2955.968 * math.cos(r(2 * D)) - 569.925 * math.cos(r(2 * Mp))
    eps = obliquity_of_ecliptic(T)
    lam = math.radians(lon)
    beta = math.radians(lat)
    eps_rad = math.radians(eps)
    x = math.cos(beta) * math.cos(lam)
    y = math.cos(beta) * math.sin(lam)
    z = math.sin(beta)
    x_eq = x
    y_eq = y * math.cos(eps_rad) - z * math.sin(eps_rad)
    z_eq = y * math.sin(eps_rad) + z * math.cos(eps_rad)
    ra = math.degrees(math.atan2(y_eq, x_eq)) % 360.0
    dec = math.degrees(math.atan2(z_eq, math.sqrt(x_eq * x_eq + y_eq * y_eq)))
    return {"lon": lon, "lat": lat, "distance_km": distance, "ra_deg": ra, "dec_deg": dec, "jd": jd}


def altitude_of_moon(dt: datetime.datetime, lat_deg: float, lon_deg: float) -> float:
    mp = moon_position(dt)
    jd = mp["jd"]
    lst = lst_deg(jd, lon_deg)
    ha = (lst - mp["ra_deg"] + 360.0) % 360.0
    if ha > 180.0:
        ha -= 360.0
    lat_r = math.radians(lat_deg)
    dec_r = math.radians(mp["dec_deg"])
    ha_r = math.radians(ha)
    sin_alt = math.sin(lat_r) * math.sin(dec_r) + math.cos(lat_r) * math.cos(dec_r) * math.cos(ha_r)
    sin_alt = max(-1.0, min(1.0, sin_alt))
    return math.degrees(math.asin(sin_alt))


def _find_crossing(start_dt: datetime.datetime, step_hours: float, func, target: float, rising: bool = True, max_hours: float = 48, tol_seconds: int = 30) -> Optional[datetime.datetime]:
    t0 = start_dt
    v0 = func(t0) - target
    step = datetime.timedelta(hours=step_hours)
    t = t0
    for _ in range(int(max_hours / step_hours)):
        t1 = t + step
        v1 = func(t1) - target
        if rising:
            if v0 <= 0 and v1 >= 0:
                a, b = t, t1
                for _ in range(60):
                    mid = a + (b - a) / 2
                    vm = func(mid) - target
                    if abs((b - a).total_seconds()) <= tol_seconds:
                        return mid.replace(tzinfo=datetime.timezone.utc)
                    if vm >= 0:
                        b = mid
                    else:
                        a = mid
                return (a + (b - a) / 2).replace(tzinfo=datetime.timezone.utc)
        else:
            if v0 >= 0 and v1 <= 0:
                a, b = t, t1
                for _ in range(60):
                    mid = a + (b - a) / 2
                    vm = func(mid) - target
                    if abs((b - a).total_seconds()) <= tol_seconds:
                        return mid.replace(tzinfo=datetime.timezone.utc)
                    if vm <= 0:
                        b = mid
                    else:
                        a = mid
                return (a + (b - a) / 2).replace(tzinfo=datetime.timezone.utc)
        t = t1
        v0 = v1
    return None


def next_rising(start_dt: datetime.datetime, lat_deg: float, lon_deg: float) -> Optional[datetime.datetime]:
    target_alt = -0.125
    return _find_crossing(start_dt, 1.0, lambda t: altitude_of_moon(t, lat_deg, lon_deg), target_alt, rising=True)


def next_setting(start_dt: datetime.datetime, lat_deg: float, lon_deg: float) -> Optional[datetime.datetime]:
    target_alt = -0.125
    return _find_crossing(start_dt, 1.0, lambda t: altitude_of_moon(t, lat_deg, lon_deg), target_alt, rising=False)


def next_transit(start_dt: datetime.datetime, lat_deg: float, lon_deg: float) -> Optional[datetime.datetime]:
    def ha_deg(t: datetime.datetime) -> float:
        mp = moon_position(t)
        return ((lst_deg(mp["jd"], lon_deg) - mp["ra_deg"] + 540.0) % 360.0) - 180.0

    return _find_crossing(start_dt, 1.0, ha_deg, 0.0, rising=False)


def main() -> None:
    # runtime entrypoint: compute events for the configured day
    timesec = time.time()
    date = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=day_offset)

    # Zodiac labels (customize as desired)
    zodiac = [
        'Walley', 'Flathead Catfish', 'Trout', 'Largemouth Bass',
        'Northern Pike', 'Smallmouth Bass', 'Bluegill', 'Crappie',
        'Striper', 'Muskie', 'Channel Catfish', 'Bowfin'
    ]

    segments = sunlight(Latitude, Longitude, timesec)
    dayhours = segments[0] / 6
    nighthours = (1440 - segments[0]) / 6
    d1 = segments[1] * 24 * 60

    # build 'zodiac' times (12 slots)
    times: List[str] = []
    for x in range(3):
        m1 = d1 - (nighthours * (3 - x))
        m1 = abs(m1) / 60 / 24
        times.append(hrmn(m1))
    for x in range(6):
        dz = d1 + (dayhours * x)
        dz = dz / 60 / 24
        times.append(hrmn(dz))
    dz = d1 + (dayhours * 6)
    for x in range(3):
        m1 = dz + (nighthours * x)
        m1 = m1 / 60 / 24
        times.append(hrmn(m1))

    # compute moon events (UTC naive datetimes)
    moonrise = next_rising(date, Latitude, Longitude)
    moonset = next_setting(date, Latitude, Longitude)
    moon_transit = next_transit(date, Latitude, Longitude)
    moon_underfoot = (moon_transit + datetime.timedelta(hours=12)) if moon_transit else None

    # Sun times: build timezone-aware local datetimes from fractional-day values
    local_ref_date = date.astimezone(LOCAL_TZ).date()
    noon_frac = segments[2]
    strans_local = frac_day_to_local_dt(noon_frac, local_ref_date, tz=LOCAL_TZ)
    sun_transit = strans_local.astimezone(datetime.timezone.utc)
    sun_underfoot = sun_transit + datetime.timedelta(hours=12)

    sunrise_frac = segments[1]
    sunset_frac = segments[3]
    sr_local_dt = frac_day_to_local_dt(sunrise_frac, local_ref_date, tz=LOCAL_TZ)
    ss_local_dt = frac_day_to_local_dt(sunset_frac, local_ref_date, tz=LOCAL_TZ)

    mr_local = to_local(moonrise)
    ms_local = to_local(moonset)
    mtrans_local = to_local(moon_transit)
    mund_local = to_local(moon_underfoot)
    sund_local = to_local(sun_underfoot)
    strans_local = to_local(sun_transit)

    windows: List[Tuple[datetime.datetime, datetime.datetime]] = []
    windows.append(make_window(mtrans_local, MAJOR_HOURS, MAJOR_HOURS))
    windows.append(make_window(mund_local, MAJOR_HOURS, MAJOR_HOURS))
    windows.append(make_window(mr_local, MINOR_HOURS, MINOR_HOURS))
    windows.append(make_window(ms_local, MINOR_HOURS, MINOR_HOURS))
    windows.append(make_window(strans_local, MINOR_HOURS, MINOR_HOURS))
    windows.append(make_window(sund_local, MINOR_HOURS, MINOR_HOURS))
    windows.append(make_window(sr_local_dt, MAJOR_HOURS, MAJOR_HOURS))
    windows.append(make_window(ss_local_dt, MAJOR_HOURS, MAJOR_HOURS))

    merged = merge_windows(windows)

    event_labels = [
        'Moon Overhead (Major)', 'Moon Underfoot (Major)',
        'Moonrise (Minor)', 'Moonset (Minor)',
        'Sun Overhead (Minor)', 'Sun Underfoot (Minor)',
        'Sunrise (Major)', 'Sunset (Major)'
    ]

    print(date, "UTC time")

    print("\nIndividual Solunar Windows (local time):")
    for lbl, (st, en) in zip(event_labels, windows):
        st_str = st.strftime('%m/%d/%Y %I:%M%p') if st else "--"
        en_str = en.strftime('%I:%M%p') if en else "--"
        print(f"{lbl:25} → {st_str} - {en_str}")

    print("\nZodiac replaced with fish:")
    if len(times) != len(zodiac):
        print(f"Warning: times length ({len(times)}) != zodiac length ({len(zodiac)}). Using modulo mapping.")

    if not zodiac:
        raise ValueError("zodiac list is empty; please provide at least one zodiac label")

    for i, t in enumerate(times):
        lbl = zodiac[i % len(zodiac)]
        print(f"{lbl:16} → {t}")

    print("\nSolunar Periods (local time):")
    print(f"Sunrise        → {fmt(sr_local_dt)}")
    print(f"Noon           → {fmt(strans_local)}")
    print(f"Sunset         → {fmt(ss_local_dt)}")
    print(f"Sun Underfoot  → {fmt(sund_local)}")
    print(f"Moonrise       → {fmt(mr_local)}")
    print(f"Moonset        → {fmt(ms_local)}")
    print(f"Moon Overhead  → {fmt(mtrans_local)}")
    print(f"Moon Underfoot → {fmt(mund_local)}")

    # Build zodiac segment datetimes (local) covering the date range spanned by merged windows
    if merged:
        min_date = min(w[0].date() for w in merged)
        max_date = max(w[1].date() for w in merged)
        start_day = min_date - datetime.timedelta(days=1)
        end_day = max_date + datetime.timedelta(days=1)
    else:
        center = datetime.datetime.now(LOCAL_TZ).date()
        start_day = center - datetime.timedelta(days=1)
        end_day = center + datetime.timedelta(days=1)

    segment_points: List[Tuple[datetime.datetime, str]] = []
    day = start_day
    while day <= end_day:
        for i, tstr in enumerate(times):
            suffix = tstr[-2:].upper() if len(tstr) > 2 else None
            if suffix in ("AM", "PM"):
                hh_mm = tstr[:-2]
                hh, mm = map(int, hh_mm.split(':'))
                if hh == 12:
                    hour24 = 0 if suffix == 'AM' else 12
                else:
                    hour24 = hh if suffix == 'AM' else hh + 12
                segment_points.append((datetime.datetime(day.year, day.month, day.day, hour24, mm, tzinfo=LOCAL_TZ), zodiac[i % len(zodiac)]))
            else:
                hh, mm = map(int, tstr.split(':'))
                segment_points.append((datetime.datetime(day.year, day.month, day.day, hh, mm, tzinfo=LOCAL_TZ), zodiac[i % len(zodiac)]))
        day += datetime.timedelta(days=1)

    segment_points.sort(key=lambda x: x[0])
    segments_zodiac: List[Tuple[datetime.datetime, datetime.datetime, str]] = []
    for i in range(len(segment_points) - 1):
        s_start, label = segment_points[i]
        s_end, _ = segment_points[i + 1]
        segments_zodiac.append((s_start, s_end, label))

    print("\nSuggested Best Fishing Windows (fish for Zodiac):")
    print(f"{'Start Time':<20} {'End Time':<10} {'Fish':<42} {'Events':<42}")
    print("-" * 105)

    for wstart, wend in merged:
        contributing_labels = [
            lbl for lbl, (orig_start, orig_end) in zip(event_labels, windows)
            if orig_start and orig_start < wend and orig_end and orig_end > wstart
        ]
        zodiac_labels: List[str] = []
        for s_start, s_end, label in segments_zodiac:
            if s_start <= wend and s_end >= wstart:
                if not zodiac_labels or zodiac_labels[-1] != label:
                    zodiac_labels.append(label)
        label_str = ", ".join(zodiac_labels) if zodiac_labels else "None"
        event_str = ", ".join(contributing_labels) if contributing_labels else "Unknown"
        print(f"{wstart.strftime('%m/%d/%Y %I:%M%p'):<19}- {wend.strftime('%I:%M%p'):<10} {label_str:<42} {event_str:<42}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute Solunar fishing windows for a location/date.")
    parser.add_argument('--lat', type=float, help='Latitude in decimal degrees (positive north)')
    parser.add_argument('--lon', type=float, help='Longitude in decimal degrees (positive east)')
    parser.add_argument('--tz', type=float, help='Timezone offset from UTC in hours (e.g. -4)')
    parser.add_argument('--day-offset', type=int, help='Integer number of days to shift the report (0 for today)')
    parser.add_argument('--date', type=str, help='Explicit date in YYYY-MM-DD format (overrides day-offset)')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    args = parser.parse_args()

    # Apply CLI overrides to module-level configuration where provided.
    # We update the timezone and LOCAL_TZ globals if tz is supplied so downstream
    # functions that reference LOCAL_TZ will use the requested timezone.
    if args.lat is not None:
        Latitude = args.lat
    if args.lon is not None:
        Longitude = args.lon
    if args.tz is not None:
        # update globals for timezone usage
        timezone = args.tz
        LOCAL_TZ = datetime.timezone(datetime.timedelta(hours=int(timezone)))
        TZ_NAME = f"UTC{int(timezone):+d}"
    if args.day_offset is not None:
        day_offset = args.day_offset
    if args.date:
        try:
            # If a specific date is supplied, set day_offset so main computes for that date
            parsed = datetime.datetime.strptime(args.date, '%Y-%m-%d').date()
            # compute a day_offset relative to today (UTC)
            today_utc = datetime.datetime.now(datetime.timezone.utc).date()
            day_offset = (parsed - today_utc).days
        except ValueError:
            parser.error('date must be YYYY-MM-DD')

    # If the user asked for debug, keep prints; otherwise they remain as-is.
    main()
# run it with custom locations without edit
# python "/Solunar Zodiac.py" --lat 36.987 --lon -119.700 --tz -7 --date 2025-10-30
