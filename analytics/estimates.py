import numpy as np
import pandas as pd


def valid_number(value):
    try:
        if value is None or pd.isna(value):
            return np.nan

        return float(value)

    except Exception:
        return np.nan


def first_available(mapping, keys):
    if not isinstance(mapping, dict):
        return np.nan

    for key in keys:
        if key in mapping:
            value = valid_number(
                mapping.get(key)
            )

            if not pd.isna(value):
                return value

    return np.nan


def calculate_estimates(
    info,
    analyst_targets,
    current_price
):

    low = first_available(
        analyst_targets,
        ["low", "lowPrice"]
    )

    mean = first_available(
        analyst_targets,
        ["mean", "meanPrice"]
    )

    median = first_available(
        analyst_targets,
        ["median", "medianPrice"]
    )

    high = first_available(
        analyst_targets,
        ["high", "highPrice"]
    )

    # Fallback to info endpoint

    if pd.isna(low):
        low = valid_number(
            info.get("targetLowPrice")
        )

    if pd.isna(mean):
        mean = valid_number(
            info.get("targetMeanPrice")
        )

    if pd.isna(median):
        median = valid_number(
            info.get("targetMedianPrice")
        )

    if pd.isna(high):
        high = valid_number(
            info.get("targetHighPrice")
        )

    analysts = valid_number(
        info.get(
            "numberOfAnalystOpinions"
        )
    )

    recommendation = info.get(
        "recommendationKey",
        "N/A"
    )

    upside_mean = np.nan

    if (
        not pd.isna(mean)
        and current_price
        and current_price > 0
    ):
        upside_mean = (
            mean / current_price - 1
        )

    return {
        "low": low,
        "mean": mean,
        "median": median,
        "high": high,
        "analysts": analysts,
        "recommendation": recommendation,
        "upside_mean": upside_mean,
    }
