import ephem
import math
import time
import datetime
"""
Put your location-specific configuration here. The main script expects the following names:

- Latitude: decimal degrees (positive north)
- Longitude: decimal degrees (positive east)
- timezone: numeric offset from UTC in hours (used as a fallback)
- day_offset: integer number of days to shift the report (0 for today)
- TZ_NAME: optional IANA timezone name (e.g. "America/Chicago"). When provided, the
  main script will compute a DST-aware UTC offset automatically.

Edit the values below for your location. If you set TZ_NAME, the script will prefer it and
compute the correct offset (including DST). If TZ_NAME is None the script will fall back to
the numeric `timezone` value (or compute system local offset if available).
"""

# Example placeholders — replace with your actual location.
# Latitude = 41.8781    # example: Chicago latitude
# Longitude = -87.6298  # example: Chicago longitude
Latitude = 36.6277
Longitude = -119.6766

# Numeric timezone offset (hours) — kept for backward compatibility / fallback.
# Example: -6 for CST (standard time). The main script will override this with a
# DST-aware offset if TZ_NAME or system timezone is available.
timezone = -7

# day_offset: 0 for today, +1 for tomorrow, -1 for yesterday
day_offset = 0

# Optional IANA timezone name. If you set this to a string like "America/Chicago",
# the main script will compute the correct UTC offset (including DST) automatically.
# Set to None if you don't want to provide an IANA name.
TZ_NAME = "America/Los_Angeles"

# Ensure `timezone` reflects the correct current UTC offset (hours), including DST.
# If the optional TZ_NAME is provided in location_config (IANA name like "America/Chicago"),
# use that to compute the proper offset; otherwise fall back to the system local offset.

try:
    # try to read an optional TZ_NAME from location_config
    from location_config import TZ_NAME
except Exception:
    TZ_NAME = None

try:
    if TZ_NAME:
        # Use IANA timezone to compute correct offset including DST
        try:
            from zoneinfo import ZoneInfo
            tz_obj = ZoneInfo(TZ_NAME)
            timezone = datetime.datetime.now(tz_obj).utcoffset().total_seconds() / 3600
        except Exception:
            # zoneinfo may not be available or TZ_NAME may be invalid; fall back to system offset
            timezone = datetime.datetime.now().astimezone().utcoffset().total_seconds() / 3600
    else:
        # No TZ_NAME provided: use the system local timezone offset (may reflect DST)
        timezone = datetime.datetime.now().astimezone().utcoffset().total_seconds() / 3600
except Exception:
    # If anything goes wrong, keep the timezone value imported from location_config
    pass

def hrmn(time):
    # return 12-hour time with AM/PM suffix (e.g. "01:05PM")
    # Work in integer minutes to avoid repeated float operations and
    # ensure consistent rounding.
    t = time % 1
    total_minutes = int(round(t * 24 * 60))
    # normalize in case rounding pushed us to 24*60
    total_minutes %= 24 * 60
    hour = total_minutes // 60
    minute = total_minutes % 60
    ampm = "AM" if hour < 12 else "PM"
    hour12 = hour % 12 or 12
    return "{:02d}:{:02d}{}".format(hour12, minute, ampm)

# after this used ephem for moon stuff...
def sunlight(Latitude, Longitude, timesec):
    # small performance: cache math helpers locally
    rad = math.radians
    deg = math.degrees
    sin = math.sin
    cos = math.cos
    tan = math.tan

    E2 = 0.1 / 24
    JD = (timesec / 86400.0) + 2440587.5 + E2 - timezone / 24
    JC = (JD - 2451545) / 36525

    GMLS = 280.46646 + JC * (36000.76983 + JC * 0.0003032)
    GMLS %= 360
    GMAS = 357.52911 + JC * (35999.05029 - 0.0001537 * JC)

    # Sun equation of center (SEC)
    SEC = (sin(rad(GMAS)) * (1.914602 - JC * (0.004817 + 0.000014 * JC))
           + sin(rad(2 * GMAS)) * (0.019993 - 0.000101 * JC)
           + sin(rad(3 * GMAS)) * 0.000289)

    STL = GMLS + SEC
    SAL = STL - 0.00569 - 0.00478 * sin(rad(125.04 - 1934.136 * JC))

    MOE = 23 + (26 + ((21.448 - JC * (46.815 + JC * (0.00059 - JC * 0.001813)))) / 60) / 60
    OC = MOE + 0.00256 * cos(rad(125.04 - 1934.136 * JC))

    SD = deg(math.asin(sin(rad(OC)) * sin(rad(SAL))))

    # solar zenith angle correction for sunrise/sunset
    sunrised = deg(math.acos(cos(rad(90.833)) / (cos(rad(Latitude)) * cos(rad(SD))) - tan(rad(Latitude)) * tan(rad(SD))))

    EEO = round(0.016708634 - JC * (0.000042037 + 0.0000001267 * JC), 2)
    vary = round(tan(rad(OC / 2)) * tan(rad(OC / 2)), 2)

    ET = 4 * deg(
        vary * sin(2 * rad(GMLS))
        - 2 * EEO * sin(rad(GMAS))
        + 4 * EEO * vary * sin(rad(GMAS)) * cos(2 * rad(GMLS))
        - 0.5 * vary * vary * sin(4 * rad(GMLS))
        - 1.25 * EEO * EEO * sin(2 * rad(GMAS))
    )

    noon = (720 - 4 * Longitude - ET + timezone * 60) / 1440
    sunrise = noon - sunrised * 4 / 1440
    sunset = noon + sunrised * 4 / 1440
    duration = 8 * sunrised
    return [duration, sunrise, noon, sunset]

def frac_day_to_local_dt(frac, ref_date):
    # frac is fraction of day [0..1) where 0 = 00:00 local
    total_seconds = frac * 24 * 3600
    # normalize in case of small rounding outside 0..24h
    total_seconds = total_seconds % (24 * 3600)
    hh = int(total_seconds // 3600)
    mm = int((total_seconds % 3600) // 60)
    ss = int(total_seconds % 60)
    return datetime.datetime(ref_date.year, ref_date.month, ref_date.day, hh, mm, ss)

# Solunar Fishing Periods (with best-time ranges)
# convert ephem datetimes (UTC) to local time using timezone offset
def to_local(dt_utc):
    return dt_utc + datetime.timedelta(hours=timezone)

def fmt(dt):
    # use 12-hour time with AM/PM for local display
    return dt.strftime('%m/%d/%Y %I:%M%p')

def make_window(center_dt, hours_before, hours_after):
    return (center_dt - datetime.timedelta(hours=hours_before), center_dt + datetime.timedelta(hours=hours_after))

def merge_windows(windows):
    if not windows:
        return []
    # sort by start
    windows = sorted(windows, key=lambda w: w[0])
    merged = [windows[0]]
    for start, end in windows[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            # overlap -> extend
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged

timesec = time.time()
date = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=day_offset)

# Zodiac labels
# old_zodiac = ['Rat','Ox','Tiger','Rabbit','Dragon','Snake','Horse','Sheep','Monkey','Bird','Dog','Boar']
zodiac = ['Walley','Flathead Catfish','Trout','Largemouth Bass',
          'Northern Pike','Smallmouth Bass','Bluegill','Crappie',
          'Striper','Muskie','Channel Catfish','Bowfin']

# Get solar segments
segments = sunlight(Latitude, Longitude, timesec)
dayhours = segments[0] / 6
nighthours = (1440 - segments[0]) / 6
d1 = segments[1] * 24 * 60

# Zodiac time slots
times = []
for x in range(3):
    m1 = d1 - (nighthours * (3 - x))
    m1 = abs(m1) / 60 / 24
    times += [hrmn(m1)]
for x in range(6):
    dz = d1 + (dayhours * x)
    dz = dz / 60 / 24
    times += [hrmn(dz)]
dz = d1 + (dayhours * 6)
for x in range(3):
    m1 = dz + (nighthours * x)
    m1 = m1 / 60 / 24
    times += [hrmn(m1)]

# Get moon events
observer = ephem.Observer()
observer.lat = str(Latitude)
observer.lon = str(Longitude)
observer.date = date

moon = ephem.Moon(observer)
moonrise = observer.next_rising(moon).datetime()
moonset = observer.next_setting(moon).datetime()
moon_transit = observer.next_transit(moon).datetime()
moon_underfoot = moon_transit + datetime.timedelta(hours=12)
# Sun events: transit (overhead) and underfoot (transit + 12h)
sun = ephem.Sun(observer)
sun_transit = observer.next_transit(sun).datetime()
sun_underfoot = sun_transit + datetime.timedelta(hours=12)

# compute sunrise/sunset local datetimes from fractional-day values returned by sunlight()
local_ref_date = (date + datetime.timedelta(hours=timezone)).date()
sunrise_frac = segments[1]
sunset_frac = segments[3]
sr_local_dt = frac_day_to_local_dt(sunrise_frac, local_ref_date)
ss_local_dt = frac_day_to_local_dt(sunset_frac, local_ref_date)

# define major/minor window sizes (hours)
MAJOR_HOURS = 1.0    # around moon transit/underfoot
MINOR_HOURS = 0.5    # around moonrise/moonset

# convert moon event times to local
mr_local = to_local(moonrise)
ms_local = to_local(moonset)
mtrans_local = to_local(moon_transit)
mund_local = to_local(moon_underfoot)
sund_local = to_local(sun_underfoot)
strans_local = to_local(sun_transit)

# build individual windows
windows = []
windows.append(make_window(mtrans_local, MAJOR_HOURS, MAJOR_HOURS))   # Moon overhead (major)
windows.append(make_window(mund_local, MAJOR_HOURS, MAJOR_HOURS))     # Moon underfoot (major)
windows.append(make_window(mr_local, MINOR_HOURS, MINOR_HOURS))       # Moonrise (minor)
windows.append(make_window(ms_local, MINOR_HOURS, MINOR_HOURS))       # Moonset (minor)
windows.append(make_window(strans_local, MINOR_HOURS, MINOR_HOURS))   # Sun overhead (minor)
windows.append(make_window(sund_local, MINOR_HOURS, MINOR_HOURS))     # Sun underfoot (minor)
windows.append(make_window(sr_local_dt, MAJOR_HOURS, MAJOR_HOURS))       # Sunrise (major)
windows.append(make_window(ss_local_dt, MAJOR_HOURS, MAJOR_HOURS))       # Sunset (major)

merged = merge_windows(windows)

# Labels for the original windows so we can show which events produced each window
event_labels = [
    'Moon Overhead (Major)', 'Moon Underfoot (Major)',
    'Moonrise (Minor)', 'Moonset (Minor)',
    'Sun Overhead (Minor)', 'Sun Underfoot (Minor)',
    'Sunrise (Major)', 'Sunset (Major)'
]

# Display fusion
print(date, "UTC time")

print("\nIndividual Solunar Windows (local time):")
for lbl, (st, en) in zip(event_labels, windows):
    print(f"{lbl:15} → {st.strftime('%m/%d/%Y %I:%M%p')} - {en.strftime('%I:%M%p')}")

print("\nZodiac replaced with fish:")
# If somebody customized `zodiac` make sure we warn about mismatches between
# the number of zodiac labels and the number of time slots (always 12).
if len(times) != len(zodiac):
    print(f"Warning: times length ({len(times)}) != zodiac length ({len(zodiac)}). Using modulo mapping.")

# Print each time and the corresponding zodiac label (wrap labels with modulo
# in case the user supplied fewer or more labels than time slots).
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


# Build zodiac segment datetimes (local) covering the date range spanned by
# the merged windows (plus one day on each side) so windows that fall on adjacent
# dates (e.g. Sun Underfoot at 00:12 next day) are included.
if merged:
    # compute earliest and latest dates from merged windows
    min_date = min(w[0].date() for w in merged)
    max_date = max(w[1].date() for w in merged)
    start_day = min_date - datetime.timedelta(days=1)
    end_day = max_date + datetime.timedelta(days=1)
else:
    # fallback: use current UTC date (module-level datetime) adjusted by timezone
    center = (datetime.datetime.utcnow() + datetime.timedelta(hours=timezone)).date()
    start_day = center - datetime.timedelta(days=1)
    end_day = center + datetime.timedelta(days=1)

segment_points = []
day = start_day
while day <= end_day:
    for i, tstr in enumerate(times):
        # handle AM/PM produced by hrmn()
        suffix = tstr[-2:].upper() if len(tstr) > 2 else None
        if suffix in ('AM', 'PM'):
            hh_mm = tstr[:-2]
            hh, mm = map(int, hh_mm.split(':'))
            # convert 12-hour to 24-hour
            if hh == 12:
                hour24 = 0 if suffix == 'AM' else 12
            else:
                hour24 = hh if suffix == 'AM' else hh + 12
            segment_points.append((datetime.datetime(day.year, day.month, day.day, hour24, mm), zodiac[i % len(zodiac)]))
        else:
            # fallback for legacy "HH:MM" format
            hh, mm = map(int, tstr.split(':'))
            segment_points.append((datetime.datetime(day.year, day.month, day.day, hh, mm), zodiac[i % len(zodiac)]))
    day += datetime.timedelta(days=1)

# sort and build consecutive segments (start, end, label)
segment_points.sort(key=lambda x: x[0])
segments_zodiac = []
for i in range(len(segment_points) - 1):
    s_start, label = segment_points[i]
    s_end, _ = segment_points[i + 1]
    segments_zodiac.append((s_start, s_end, label))

print("\nSuggested Best Fishing Windows (fish for Zodiac):")
print(f"{'Start Time':<20} {'End Time':<10} {'Events':<42} {'Fish':<30}")
print("-" * 105)

for wstart, wend in merged:
    # Find contributing original labels
    contributing_labels = [
        lbl for lbl, (orig_start, orig_end) in zip(event_labels, windows)
        if orig_start < wend and orig_end > wstart
    ]
    # Find overlapping zodiac segments
    zodiac_labels = []
    for s_start, s_end, label in segments_zodiac:
        # Use inclusive overlap test so segments that exactly touch a window
        # boundary are considered overlapping. Strict (<, >) could miss
        # cases where s_end == wstart or s_start == wend and produce "None".
        if s_start <= wend and s_end >= wstart:
            if not zodiac_labels or zodiac_labels[-1] != label:
                zodiac_labels.append(label)
    label_str = ", ".join(zodiac_labels) if zodiac_labels else "None"
    event_str = ", ".join(contributing_labels) if contributing_labels else "Unknown"
    print(f"{wstart.strftime('%m/%d/%Y %I:%M%p'):<20} {wend.strftime('%I:%M%p'):<10} {event_str:<42} {label_str:<30}")
