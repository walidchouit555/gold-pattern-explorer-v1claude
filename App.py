import requests
import streamlit as st
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ==================================================
# 1. Page configuration
# ==================================================

st.set_page_config(
    page_title="Gold Pattern Explorer",
    layout="wide"
)

st.title("🟠 Gold Pattern Explorer")
st.write("Historical Gold Pattern Matching & Backtesting")
st.info(
    "يجلب التطبيق بيانات الذهب تلقائيًا عند الإمكان، مع الاحتفاظ بإمكانية "
    "رفع ملف CSV يدويًا كخيار بديل."
)

API_URL = "https://xaus.com/api/v1/chart"
MIN_HISTORY_MONTHS = 6
MAX_LATEST_CANDLE_AGE_HOURS = 72


# ==================================================
# 2. Data helpers
# ==================================================

def validate_history(dataframe):
    """Normalize OHLC history and reject invalid or insufficient data."""
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise ValueError("لم تصل أي شموع صالحة من مصدر البيانات.")

    history = dataframe.copy()
    history.columns = [str(column).strip().lower() for column in history.columns]

    # Accept both CSV names and XAUS API's compact OHLCV names.
    history = history.rename(columns={
        "t": "timestamp",
        "time": "timestamp",
        "datetime": "timestamp",
        "date": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    })

    required_columns = ["timestamp", "open", "high", "low", "close"]
    missing_columns = [column for column in required_columns if column not in history.columns]
    if missing_columns:
        raise ValueError(
            "الأعمدة المطلوبة غير موجودة: " + ", ".join(missing_columns)
        )

    history["timestamp"] = pd.to_datetime(
        history["timestamp"], utc=True, errors="coerce"
    )

    price_columns = ["open", "high", "low", "close"]
    for column in price_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")

    if "volume" in history.columns:
        history["volume"] = pd.to_numeric(history["volume"], errors="coerce")
    else:
        history["volume"] = np.nan

    history = history.dropna(subset=["timestamp"] + price_columns)
    history = history.sort_values("timestamp")
    history = history.drop_duplicates(subset=["timestamp"], keep="last")

    valid_ohlc = (
        (history["high"] >= history["low"])
        & (history["high"] >= history["open"])
        & (history["high"] >= history["close"])
        & (history["low"] <= history["open"])
        & (history["low"] <= history["close"])
        & (history[price_columns] > 0).all(axis=1)
    )
    history = history.loc[valid_ohlc].reset_index(drop=True)

    if history.empty:
        raise ValueError("لم تبقَ أي شموع صالحة بعد التحقق من الأسعار.")

    # Require at least six calendar months between the first and last candles.
    first_timestamp = history["timestamp"].min()
    last_timestamp = history["timestamp"].max()
    six_months_before_last = last_timestamp - pd.DateOffset(months=MIN_HISTORY_MONTHS)
    if first_timestamp > six_months_before_last:
        raise ValueError(
            "الفترة التاريخية المستلمة أقل من ستة أشهر. "
            f"الفترة المتاحة: {first_timestamp} — {last_timestamp}."
        )

    # The latest candle must not be excessively old. A 72-hour threshold allows
    # normal weekend/holiday gaps but rejects obviously stale automatic data.
    age = pd.Timestamp.now(tz="UTC") - last_timestamp
    if age > pd.Timedelta(hours=MAX_LATEST_CANDLE_AGE_HOURS):
        raise ValueError(
            "آخر شمعة قديمة جدًا للاستخدام التلقائي "
            f"(عمرها التقريبي {age.total_seconds() / 3600:.1f} ساعة)."
        )

    return history


def api_timestamp(value):
    """Parse XAUS timestamps expressed as Unix seconds/milliseconds or ISO text."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return pd.NaT
        unit = "ms" if abs(float(value)) >= 1e12 else "s"
        return pd.to_datetime(value, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


def fetch_automatic_history():
    """Fetch two years of hourly XAU/USD candles and validate the response."""
    response = requests.get(
        API_URL,
        params={"symbol": "xau", "range": "2y", "interval": "1h"},
        timeout=25,
        headers={"User-Agent": "GoldPatternExplorer/1.0"},
    )
    response.raise_for_status()
    payload = response.json()

    data_state = payload.get("data_state", {}) if isinstance(payload, dict) else {}
    api_status = str(data_state.get("status", "")).lower() if isinstance(data_state, dict) else ""
    if api_status == "unavailable":
        raise ValueError("مصدر البيانات يعلن أن البيانات غير متاحة حاليًا.")
    if api_status == "stale":
        raise ValueError("مصدر البيانات يعلن أن البيانات التاريخية قديمة؛ لم تُستخدم تلقائيًا.")

    if isinstance(payload, list):
        points = payload
    elif isinstance(payload, dict):
        points = None
        for key in ("points", "data", "candles", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                points = value
                break
        if points is None:
            # Handle a standard chart-response wrapper if the upstream changes shape.
            chart = payload.get("chart")
            if isinstance(chart, dict):
                result = chart.get("result")
                if isinstance(result, list) and result and isinstance(result[0], dict):
                    timestamps = result[0].get("timestamp", [])
                    quote = (result[0].get("indicators", {}).get("quote") or [{}])[0]
                    points = [
                        {
                            "t": timestamps[i],
                            "o": (quote.get("open") or [None] * len(timestamps))[i],
                            "h": (quote.get("high") or [None] * len(timestamps))[i],
                            "l": (quote.get("low") or [None] * len(timestamps))[i],
                            "c": (quote.get("close") or [None] * len(timestamps))[i],
                            "v": (quote.get("volume") or [None] * len(timestamps))[i],
                        }
                        for i in range(len(timestamps))
                    ]
        if points is None:
            raise ValueError("صيغة استجابة مصدر البيانات غير معروفة.")
    else:
        raise ValueError("استجابة مصدر البيانات ليست بصيغة JSON المتوقعة.")

    if not points:
        raise ValueError("مصدر البيانات لم يُرجع أي شموع.")

    raw = pd.DataFrame(points)
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    timestamp_column = next(
        (column for column in ("t", "timestamp", "time", "datetime", "date") if column in raw.columns),
        None,
    )
    if timestamp_column is None:
        raise ValueError("استجابة مصدر البيانات لا تحتوي على وقت الشمعة.")

    raw["timestamp"] = raw[timestamp_column].map(api_timestamp)
    raw = raw.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    if timestamp_column != "timestamp" and timestamp_column in raw.columns:
        # The parsed timestamp column above is authoritative.
        raw = raw.drop(columns=[timestamp_column])

    history = validate_history(raw)
    return history, data_state


def load_csv_history(uploaded_file):
    """Read an uploaded CSV using the same validation rules as automatic data."""
    csv_data = pd.read_csv(uploaded_file)
    return validate_history(csv_data)


# ==================================================
# 2b. Walk-forward backtest helpers
# ==================================================

# BACKTEST HELPERS START
def build_shape_windows(close, length):
    """Row s = close[s:s+length] expressed as % change from its first candle.

    This is the same normalisation used by the matching section, so a
    backtest "sees" patterns exactly the way the explorer does.
    """
    windows = sliding_window_view(np.asarray(close, dtype=float), length)
    return windows / windows[:, :1] - 1.0


def pick_top_matches(rmse, spacing, top_k):
    """Best matches by lowest RMSE, skipping near-duplicates.

    Neighbouring windows are almost identical, so without a minimum spacing
    the "top 5" would be five copies of the same market episode.
    """
    chosen = []
    for idx in np.argsort(rmse, kind="stable"):
        idx = int(idx)
        if all(abs(idx - other) >= spacing for other in chosen):
            chosen.append(idx)
            if len(chosen) == top_k:
                break
    return chosen


def run_walk_forward(
    history,
    pattern_length,
    horizon,
    top_k,
    threshold,
    cost,
    warmup_start,
    step,
    progress_callback=None,
):
    """Walk-forward test: at each decision time use ONLY data known by then.

    Decision time t (exclusive end of the target pattern):
      target pattern = candles [t - L, t)
      entry          = close of candle t - 1
      exit           = close of candle t - 1 + horizon
    Candidate patterns must have their own outcome fully known before t,
    and must not overlap the target. `threshold` and `cost` are fractions.
    """
    close = history["close"].to_numpy(dtype=float)
    timestamps = history["timestamp"].tolist()
    n = len(close)

    shapes = build_shape_windows(close, pattern_length)
    sumsq = np.einsum("ij,ij->i", shapes, shapes)
    min_matches = min(3, top_k)

    decision_points = list(range(warmup_start, n - horizon + 1, step))
    rows = []

    for position, t in enumerate(decision_points):
        target_row = t - pattern_length
        last_allowed_start = t - pattern_length - max(horizon, pattern_length)

        direction = 0
        signal = np.nan
        used = 0

        if last_allowed_start >= 0:
            count = last_allowed_start + 1
            pool = shapes[:count]
            # mean((a-b)^2) = (|a|^2 - 2 a.b + |b|^2) / L  -> one matrix-vector product
            mse = (sumsq[:count] - 2.0 * (pool @ shapes[target_row]) + sumsq[target_row]) / pattern_length
            rmse = np.sqrt(np.clip(mse, 0.0, None))

            chosen = pick_top_matches(rmse, pattern_length, top_k)
            used = len(chosen)
            if used >= min_matches:
                ends = np.array(chosen) + pattern_length - 1
                forward = close[ends + horizon] / close[ends] - 1.0
                signal = float(np.median(forward))
                if signal > threshold:
                    direction = 1
                elif signal < -threshold:
                    direction = -1

        entry_idx = t - 1
        exit_idx = t - 1 + horizon
        actual = close[exit_idx] / close[entry_idx] - 1.0
        net = direction * actual - cost if direction != 0 else 0.0

        rows.append({
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "entry_time": timestamps[entry_idx],
            "exit_time": timestamps[exit_idx],
            "matches_used": used,
            "signal": signal,
            "direction": direction,
            "actual_return": actual,
            "net_return": net,
        })

        if progress_callback is not None and (
            position % 25 == 0 or position == len(decision_points) - 1
        ):
            progress_callback((position + 1) / len(decision_points))

    return pd.DataFrame(rows)


def max_drawdown(equity):
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def permutation_p_value(directions, actual, cost, observed_mean, n_perm=2000, seed=42):
    """How often do the same long/short/flat calls, randomly re-assigned to
    different trade slots, do at least as well as the real ones?"""
    directions = np.asarray(directions)
    actual = np.asarray(actual, dtype=float)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(directions)
        net = shuffled * actual - cost * (shuffled != 0)
        if net.mean() >= observed_mean:
            hits += 1
    return (hits + 1) / (n_perm + 1)

def select_matches(starts_best_first, spacing, count):
    """Pick up to `count` match starts, best first, at least `spacing` apart.
    spacing <= 0 disables the overlap filter."""
    chosen = []
    for start in starts_best_first:
        start = int(start)
        if spacing <= 0 or all(abs(start - other) >= spacing for other in chosen):
            chosen.append(start)
            if len(chosen) == count:
                break
    return chosen


def forward_paths(close, starts, pattern_length, horizon):
    """One row per match: close[last+k] / close[last] - 1 for k = 0..horizon,
    where `last` is the final candle of the matched pattern."""
    close = np.asarray(close, dtype=float)
    rows = []
    for start in starts:
        last = int(start) + pattern_length - 1
        if last + horizon < len(close):
            rows.append(close[last:last + horizon + 1] / close[last] - 1.0)
    if not rows:
        return np.empty((0, horizon + 1))
    return np.vstack(rows)


def base_rate_returns(close, horizon):
    """Return over `horizon` candles from every possible entry: the
    unconditional benchmark a pattern's outcomes must be compared against."""
    close = np.asarray(close, dtype=float)
    if len(close) <= horizon:
        return np.array([])
    return close[horizon:] / close[:-horizon] - 1.0
# BACKTEST HELPERS END


# ==================================================
# 3. Automatic data fetching, with manual fallback
# ==================================================

if "auto_fetch_attempted" not in st.session_state:
    st.session_state["auto_fetch_attempted"] = True
    try:
        with st.spinner("جارٍ جلب بيانات الذهب التاريخية تلقائيًا..."):
            auto_history, auto_state = fetch_automatic_history()
        st.session_state["auto_history_data"] = auto_history
        st.session_state["auto_data_state"] = auto_state
        st.session_state["auto_fetch_error"] = None
        st.session_state["source_mode"] = "Automatic data"
    except Exception as error:
        st.session_state["auto_history_data"] = None
        st.session_state["auto_data_state"] = {}
        st.session_state["auto_fetch_error"] = str(error)
        st.session_state["source_mode"] = "Upload CSV"

st.header("1. Data Source")

if st.button("🔄 Refresh automatic gold data"):
    try:
        with st.spinner("جارٍ تحديث بيانات الذهب التاريخية..."):
            auto_history, auto_state = fetch_automatic_history()
        st.session_state["auto_history_data"] = auto_history
        st.session_state["auto_data_state"] = auto_state
        st.session_state["auto_fetch_error"] = None
        st.session_state["source_mode"] = "Automatic data"
        st.success("تم تحديث البيانات التلقائية بنجاح.")
    except Exception as error:
        st.session_state["auto_fetch_error"] = str(error)
        st.warning(
            "تعذّر تحديث البيانات التلقائية. يمكنك متابعة العمل بملف CSV اليدوي. "
            f"التفاصيل: {error}"
        )

source_mode = st.radio(
    "Choose data source",
    options=["Automatic data", "Upload CSV"],
    horizontal=True,
    key="source_mode",
)

history = None

if source_mode == "Automatic data":
    auto_history = st.session_state.get("auto_history_data")
    if isinstance(auto_history, pd.DataFrame) and not auto_history.empty:
        history = auto_history.copy()
        st.success(
            f"تم تحميل {len(history):,} شمعة تاريخية تلقائيًا. "
            "المصدر: XAUS (XAU/USD، إطار الساعة)."
        )
        st.caption(
            "الفترة التاريخية (UTC): "
            f"{history['timestamp'].min()} — {history['timestamp'].max()}"
        )
        state = st.session_state.get("auto_data_state", {})
        if isinstance(state, dict) and state.get("as_of"):
            st.caption(f"حالة/وقت المصدر حسب API: {state.get('status', 'غير محدد')} — {state['as_of']}")
        st.dataframe(history.tail(5), use_container_width=True, hide_index=True)
    else:
        error_message = st.session_state.get("auto_fetch_error")
        st.warning(
            "البيانات التلقائية غير متاحة حاليًا. اختر «Upload CSV» "
            "للعودة إلى الطريقة القديمة، أو اضغط زر التحديث لإعادة المحاولة."
        )
        if error_message:
            st.caption(f"تفاصيل فشل الجلب التلقائي: {error_message}")
else:
    history_file = st.file_uploader(
        "Upload historical gold data (CSV)",
        type=["csv"],
        key="history",
        help="يجب أن يحتوي الملف على timestamp, open, high, low, close."
    )
    if history_file is not None:
        try:
            history = load_csv_history(history_file)
            st.session_state["csv_history_data"] = history
            st.success(f"Loaded {len(history):,} valid candles from CSV.")
            st.caption(
                "الفترة التاريخية (UTC): "
                f"{history['timestamp'].min()} — {history['timestamp'].max()}"
            )
            st.dataframe(history.tail(5), use_container_width=True, hide_index=True)
        except Exception as error:
            st.error(f"تعذرت قراءة ملف CSV أو التحقق منه: {error}")
    elif isinstance(st.session_state.get("csv_history_data"), pd.DataFrame):
        history = st.session_state["csv_history_data"].copy()
        st.info("تم الاحتفاظ بملف CSV الذي رفعته سابقًا خلال هذه الجلسة.")

if isinstance(history, pd.DataFrame):
    st.session_state["history_data"] = history.copy()
else:
    st.session_state.pop("history_data", None)


# ==================================================
# 4. Historical pattern matching (existing method retained)
# ==================================================

st.divider()
st.header("2. Historical Matching")

if "history_data" not in st.session_state:
    st.info("اختر مصدر البيانات وحمّل البيانات قبل بدء البحث عن الأنماط.")
else:
    history = st.session_state["history_data"].copy()

    pattern_length = st.selectbox(
        "Pattern length (hourly candles)",
        [48, 72, 96, 144],
        index=3,
        key="matching_pattern_length"
    )

    # Changing the source or pattern length should not reuse an old target
    # index that would move the target away from the latest candles.
    active_source = st.session_state.get("_active_data_source")
    previous_pattern_length = st.session_state.get("_previous_pattern_length")
    if active_source != source_mode or (
        previous_pattern_length is not None
        and previous_pattern_length != pattern_length
    ):
        for length in [48, 72, 96, 144]:
            st.session_state.pop(f"matching_target_start_{length}", None)
    st.session_state["_active_data_source"] = source_mode
    st.session_state["_previous_pattern_length"] = pattern_length

    minimum_candles = pattern_length * 2 + 1

    if len(history) < minimum_candles:
        st.warning(
            f"At least {minimum_candles} valid candles are needed "
            f"for a {pattern_length}-candle pattern and an earlier historical comparison."
        )
    else:
        max_start = len(history) - pattern_length
        target_start = st.number_input(
            "Target starting candle index",
            min_value=0,
            max_value=max_start,
            value=max_start,
            step=1,
            key=f"matching_target_start_{pattern_length}"
        )

        target_start = int(target_start)
        target_end = target_start + pattern_length
        target = history.iloc[target_start:target_end].copy()

        st.subheader("Selected target")
        st.write(
            "Target period (UTC):",
            str(target["timestamp"].iloc[0]),
            "—",
            str(target["timestamp"].iloc[-1])
        )
        st.caption(
            "عند اختيار آخر فهرس، تكون العينة هي أحدث الشموع المتاحة "
            "بعدد يساوي طول النمط المختار."
        )
        st.line_chart(target.set_index("timestamp")["close"])

        if len(target) != pattern_length:
            st.warning("The selected target does not contain the required number of candles.")
        else:
            target_close = target["close"].to_numpy(dtype=float)
            target_base = target_close[0]

            if not np.isfinite(target_close).all() or target_base == 0:
                st.error("The target pattern contains invalid prices.")
            else:
                target_shape = (target_close / target_base) - 1
                results = []

                # Historical candidates end before the selected target begins.
                for start in range(0, target_start - pattern_length + 1):
                    end = start + pattern_length
                    candidate = history.iloc[start:end]
                    if len(candidate) != pattern_length:
                        continue

                    candidate_close = candidate["close"].to_numpy(dtype=float)
                    if not np.isfinite(candidate_close).all():
                        continue
                    candidate_base = candidate_close[0]
                    if candidate_base == 0:
                        continue

                    candidate_shape = (candidate_close / candidate_base) - 1
                    rmse = float(np.sqrt(np.mean((target_shape - candidate_shape) ** 2)))
                    similarity = 100 / (1 + rmse * 100)
                    future = history.iloc[end:min(end + 12, len(history))].copy()

                    results.append({
                        "start": start,
                        "end": end,
                        "rmse": rmse,
                        "similarity": similarity,
                        "future": future,
                    })

                results.sort(key=lambda item: item["rmse"])
                st.subheader("Historical matching results")

                if not results:
                    st.warning("No eligible historical matches were found. Try another target starting index.")
                else:
                    st.success(f"Found {len(results):,} historical matches.")
                    top_n = st.slider(
                        "Number of matches to display",
                        min_value=1,
                        max_value=min(20, len(results)),
                        value=min(10, len(results))
                    )

                    summary = []
                    for rank, result in enumerate(results[:top_n], start=1):
                        matched = history.iloc[result["start"]:result["end"]]
                        summary.append({
                            "Rank": rank,
                            "Similarity (%)": round(result["similarity"], 2),
                            "Start (UTC)": str(matched["timestamp"].iloc[0]),
                            "End (UTC)": str(matched["timestamp"].iloc[-1]),
                        })

                    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
                    st.caption(
                        "Similarity is a mathematical shape score, not a probability or a price forecast."
                    )

                    # ------------------------------------------------------
                    # Aggregate outcome of the best matches
                    # ------------------------------------------------------
                    st.subheader("Aggregate outcome of top matches")

                    if len(results) < 4:
                        st.info("عدد التطابقات قليل جدًا لتجميع النتائج.")
                    else:
                        agg_col1, agg_col2, agg_col3 = st.columns(3)
                        with agg_col1:
                            agg_count = st.slider(
                                "Matches to aggregate",
                                min_value=3,
                                max_value=min(50, len(results)),
                                value=min(10, len(results)),
                                key="agg_count",
                            )
                        with agg_col2:
                            agg_horizon = st.number_input(
                                "Look-ahead horizon (candles)",
                                min_value=1, max_value=72, value=12, step=1,
                                key="agg_horizon",
                            )
                        with agg_col3:
                            agg_no_overlap = st.checkbox(
                                "Exclude overlapping matches",
                                value=True,
                                key="agg_no_overlap",
                                help="يمنع احتساب النافذة نفسها مرتين عبر تطابقات متجاورة.",
                            )

                        agg_horizon = int(agg_horizon)
                        close_all = history["close"].to_numpy(dtype=float)

                        # A match is usable only if its outcome was already known
                        # by the end of the target pattern (no peeking ahead).
                        eligible_starts = [
                            item["start"] for item in results
                            if item["end"] - 1 + agg_horizon <= target_end - 1
                        ]
                        chosen_starts = select_matches(
                            eligible_starts,
                            pattern_length if agg_no_overlap else 0,
                            agg_count,
                        )
                        paths = forward_paths(close_all, chosen_starts, pattern_length, agg_horizon)

                        if len(paths) < 3:
                            st.warning(
                                f"تتوفر {len(paths)} تطابقات فقط بنتائج كاملة لهذا الأفق. "
                                "قلّل الأفق أو ألغِ استبعاد التطابقات المتداخلة."
                            )
                        else:
                            final_returns = paths[:, -1]
                            base_returns = base_rate_returns(close_all[:target_end], agg_horizon)

                            g1, g2, g3, g4 = st.columns(4)
                            g1.metric("Matches used", f"{len(final_returns)}")
                            g2.metric("Up after horizon", f"{(final_returns > 0).mean() * 100:.0f}%")
                            g3.metric("Median return", f"{np.median(final_returns) * 100:.2f}%")
                            g4.metric("Mean return", f"{final_returns.mean() * 100:.2f}%")

                            g5, g6, g7, g8 = st.columns(4)
                            g5.metric("Best case", f"{final_returns.max() * 100:.2f}%")
                            g6.metric("Worst case", f"{final_returns.min() * 100:.2f}%")
                            if base_returns.size:
                                g7.metric("Baseline: up (all windows)", f"{(base_returns > 0).mean() * 100:.0f}%")
                                g8.metric("Baseline: median", f"{np.median(base_returns) * 100:.2f}%")

                            st.caption(
                                "الخط الأساسي (Baseline) هو نتيجة أي نافذة عشوائية في نفس التاريخ. "
                                "نسبة الصعود بعد التطابقات لا تعني شيئًا إلا إذا اختلفت عنه بوضوح."
                            )

                            # Distribution of outcomes
                            bin_count = int(min(12, max(4, len(final_returns) // 2)))
                            counts, edges = np.histogram(final_returns * 100, bins=bin_count)
                            centers = np.round((edges[:-1] + edges[1:]) / 2, 3)
                            hist_frame = pd.DataFrame(
                                {"Matches": counts},
                                index=pd.Index(centers, name="Return after horizon (%)"),
                            )
                            st.bar_chart(hist_frame)

                            # Typical path over the horizon
                            path_frame = pd.DataFrame(
                                {
                                    "Median": np.median(paths, axis=0) * 100,
                                    "Mean": paths.mean(axis=0) * 100,
                                    "25th percentile": np.percentile(paths, 25, axis=0) * 100,
                                    "75th percentile": np.percentile(paths, 75, axis=0) * 100,
                                },
                                index=pd.Index(range(agg_horizon + 1), name="Candles after pattern"),
                            )
                            st.line_chart(path_frame)
                            st.caption(
                                "عدد العينات صغير والتطابقات ليست مستقلة تمامًا، لذا اعتبر هذه الأرقام "
                                "وصفًا تاريخيًا لا توقعًا ولا احتمالًا."
                            )

                    selected_rank = st.selectbox(
                        "Inspect a historical match",
                        options=list(range(1, top_n + 1)),
                        format_func=lambda rank: f"Match {rank}"
                    )
                    selected = results[selected_rank - 1]
                    st.write("Similarity score:", f"{selected['similarity']:.2f}%")

                    matched = history.iloc[selected["start"]:selected["end"]].copy()
                    st.subheader("Matched historical pattern")
                    st.line_chart(matched.set_index("timestamp")["close"])

                    future = selected["future"]
                    if not future.empty:
                        st.subheader("What happened afterward")
                        st.line_chart(future.set_index("timestamp")["close"])
                        st.dataframe(
                            future[["timestamp", "open", "high", "low", "close"]],
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No subsequent candles are available after this historical match.")


# ==================================================
# 5. Walk-forward backtest
# ==================================================

st.divider()
st.header("3. Walk-Forward Backtest")
st.caption(
    "يختبر الاختبار الرجعي الطريقة كما لو كنت تستخدمها في الماضي: عند كل نقطة قرار "
    "يستخدم فقط البيانات المعروفة حتى تلك اللحظة، ثم يقارن توقعه بما حدث فعلًا."
)

if "history_data" not in st.session_state:
    st.info("حمّل البيانات أولًا لتشغيل الاختبار الرجعي.")
else:
    bt_history = st.session_state["history_data"].reset_index(drop=True)
    bt_n = len(bt_history)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        bt_length = st.selectbox(
            "Pattern length (candles)", [48, 72, 96, 144], index=3, key="bt_pattern_length"
        )
        bt_horizon = st.number_input(
            "Holding horizon (candles)", min_value=1, max_value=72, value=12, step=1,
            key="bt_horizon",
            help="كم شمعة تُحتفظ بالصفقة بعد نهاية النمط."
        )
    with col_b:
        bt_top_k = st.slider(
            "Matches used per decision", min_value=3, max_value=20, value=5, key="bt_top_k",
            help="عدد أفضل التطابقات (غير المتداخلة) التي يُؤخذ وسيط عوائدها."
        )
        bt_threshold_pct = st.number_input(
            "Signal threshold (%)", min_value=0.0, max_value=2.0, value=0.05, step=0.01,
            key="bt_threshold",
            help="إذا كان الوسيط أقل من هذا الحد بالقيمة المطلقة فلا تُفتح صفقة."
        )
    with col_c:
        bt_cost_bps = st.number_input(
            "Round-trip cost (bps)", min_value=0.0, max_value=100.0, value=5.0, step=0.5,
            key="bt_cost",
            help="تكلفة الدخول والخروج (السبريد والعمولة) لكل صفقة، بنقاط الأساس. 1 bp = 0.01%."
        )
        bt_warmup_pct = st.slider(
            "Initial history before first decision (%)", min_value=30, max_value=80, value=50,
            key="bt_warmup",
            help="الجزء الأول من البيانات يُستخدم كذاكرة فقط، ويبدأ الاختبار بعده."
        )

    bt_step = st.number_input(
        "Decision every N candles", min_value=int(bt_horizon), max_value=500,
        value=int(bt_horizon), step=1,
        help="لا يقل عن مدة الاحتفاظ حتى لا تتداخل الصفقات."
    )

    bt_warmup_start = max(int(bt_n * bt_warmup_pct / 100), 4 * bt_length)
    bt_points = max(0, (bt_n - int(bt_horizon) - bt_warmup_start) // int(bt_step) + 1)

    if bt_points < 30:
        st.warning(
            f"عدد نقاط القرار المتاحة ({bt_points}) قليل جدًا لنتيجة ذات معنى. "
            "قلّل نسبة التاريخ الابتدائي أو طول النمط أو فترة القرار."
        )
    else:
        st.caption(f"سيُجرى الاختبار على {bt_points:,} نقطة قرار.")

    bt_signature = (bt_n, str(bt_history["timestamp"].iloc[0]), str(bt_history["timestamp"].iloc[-1]))

    if st.button("▶ Run walk-forward backtest", disabled=bt_points < 1):
        progress = st.progress(0.0, text="Running walk-forward backtest...")
        try:
            trades = run_walk_forward(
                bt_history,
                pattern_length=int(bt_length),
                horizon=int(bt_horizon),
                top_k=int(bt_top_k),
                threshold=bt_threshold_pct / 100.0,
                cost=bt_cost_bps / 10000.0,
                warmup_start=bt_warmup_start,
                step=int(bt_step),
                progress_callback=lambda fraction: progress.progress(
                    min(1.0, fraction), text="Running walk-forward backtest..."
                ),
            )
            st.session_state["backtest_result"] = {
                "trades": trades,
                "signature": bt_signature,
                "cost": bt_cost_bps / 10000.0,
                "params": {
                    "Pattern length": int(bt_length),
                    "Horizon": int(bt_horizon),
                    "Matches used": int(bt_top_k),
                    "Threshold (%)": bt_threshold_pct,
                    "Cost (bps)": bt_cost_bps,
                },
            }
        except Exception as error:
            st.session_state.pop("backtest_result", None)
            st.error(f"تعذّر تشغيل الاختبار الرجعي: {error}")
        finally:
            progress.empty()

    result = st.session_state.get("backtest_result")
    if result is not None and result["signature"] != bt_signature:
        st.info("تغيّرت البيانات منذ آخر اختبار. اضغط تشغيل لإعادة الحساب.")
    elif result is not None:
        trades = result["trades"]
        cost = result["cost"]
        close_all = bt_history["close"].to_numpy(dtype=float)
        active = trades[trades["direction"] != 0]

        st.subheader("Backtest results")
        st.caption(
            "الإعدادات: " + " | ".join(f"{key}: {value}" for key, value in result["params"].items())
        )

        if active.empty:
            st.info("لم تُفتح أي صفقة بهذه الإعدادات. جرّب تقليل حد الإشارة.")
        else:
            strategy_equity = (1.0 + trades["net_return"]).cumprod()
            always_long_net = trades["actual_return"] - cost
            always_long_equity = (1.0 + always_long_net).cumprod()
            first_entry = int(trades["entry_idx"].iloc[0])
            buy_hold_curve = close_all[trades["exit_idx"].to_numpy()] / close_all[first_entry]

            strategy_total = float(strategy_equity.iloc[-1] - 1.0)
            always_long_total = float(always_long_equity.iloc[-1] - 1.0)
            buy_hold_total = float(buy_hold_curve[-1] - 1.0)
            avg_net = float(active["net_return"].mean())
            hit_rate = float((active["net_return"] > 0).mean())
            drawdown = max_drawdown(strategy_equity)

            p_value = permutation_p_value(
                trades["direction"].to_numpy(),
                trades["actual_return"].to_numpy(),
                cost,
                float(trades["net_return"].mean()),
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Trades taken", f"{len(active):,}", f"{len(trades) - len(active):,} flat", delta_color="off")
            m2.metric("Hit rate (after costs)", f"{hit_rate * 100:.1f}%")
            m3.metric("Avg net / trade", f"{avg_net * 100:.3f}%")
            m4.metric("Max drawdown", f"{drawdown * 100:.1f}%")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Strategy total", f"{strategy_total * 100:.1f}%")
            m6.metric("Always long (same schedule)", f"{always_long_total * 100:.1f}%")
            m7.metric("Buy & hold", f"{buy_hold_total * 100:.1f}%")
            m8.metric("Random-timing p-value", f"{p_value:.3f}")

            st.caption(
                "Test period (UTC): "
                f"{trades['entry_time'].iloc[0]} — {trades['exit_time'].iloc[-1]}"
            )

            equity_frame = pd.DataFrame(
                {
                    "Strategy": strategy_equity.to_numpy(),
                    "Always long (same schedule)": always_long_equity.to_numpy(),
                    "Buy & hold": buy_hold_curve,
                },
                index=pd.Index(trades["exit_time"], name="timestamp"),
            )
            st.line_chart(equity_frame)

            if p_value < 0.05 and avg_net > 0 and strategy_total > max(always_long_total, 0.0):
                st.success(
                    "على هذه العينة تفوّقت الاستراتيجية على التوقيت العشوائي وعلى خط الأساس. "
                    "هذه فترة اختبار واحدة؛ لا تعدّها دليلًا كافيًا قبل تأكيدها على بيانات أخرى."
                )
            else:
                st.warning(
                    "لا يوجد دليل موثوق على ميزة تداولية في هذه العينة: إمّا أن الأداء لا يتفوق على "
                    "الأساس أو أنه لا يختلف إحصائيًا عن التوقيت العشوائي."
                )

            with st.expander("Trade log"):
                log = trades.copy()
                log["signal (%)"] = (log["signal"] * 100).round(3)
                log["actual (%)"] = (log["actual_return"] * 100).round(3)
                log["net (%)"] = (log["net_return"] * 100).round(3)
                log["direction"] = log["direction"].map({1: "Long", -1: "Short", 0: "Flat"})
                st.dataframe(
                    log[["entry_time", "exit_time", "matches_used", "signal (%)", "direction", "actual (%)", "net (%)"]],
                    use_container_width=True,
                    hide_index=True,
                )

        st.caption(
            "ملاحظة: إذا عدّلت الإعدادات مرارًا حتى تظهر نتيجة جيدة فأنت تضبط الطريقة على هذه العينة "
            "(overfitting). حدّد الإعدادات قبل التشغيل، وجرّبها لاحقًا على فترة أو بيانات لم تُستخدم."
        )
