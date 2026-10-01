"""Safe, closed-catalog HTML rendering for the exponential MLE vertical."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from html import escape
from math import isfinite
from types import MappingProxyType
from typing import Final

from veridist.families.exponential import (
    ExponentialFit,
    ExponentialFitFailure,
    ExponentialFitFailureCode,
    ExponentialFitSuccess,
)


class ReportLocale(StrEnum):
    """The closed set of report locales supported by this vertical."""

    EN = "en"
    FA = "fa"
    DE = "de"


REPORT_KEYS: Final = (
    "status",
    "family",
    "parameterization",
    "location",
    "rate",
    "mean",
    "observation_count",
    "event_count",
    "censored_count",
    "total_time",
    "log_likelihood",
    "inference",
    "censoring_assumption",
    "failure_code",
)

REPORT_LABEL_KEYS: Final[Mapping[str, str]] = MappingProxyType(
    {key: f"report.{key}" for key in REPORT_KEYS}
)

_EN_CATALOG: Final[Mapping[str, str]] = MappingProxyType(
    {
        "status": "Exponential fit status",
        "family": "Family",
        "parameterization": "Parameterization",
        "location": "Fixed location",
        "rate": "Rate",
        "mean": "Derived mean",
        "observation_count": "Observation count",
        "event_count": "Observed event count",
        "censored_count": "Right-censored count",
        "total_time": "Total time on test",
        "log_likelihood": "Log likelihood",
        "inference": "Inference capability",
        "censoring_assumption": "Censoring assumption",
        "failure_code": "Failure code",
    }
)
_FA_CATALOG: Final[Mapping[str, str]] = MappingProxyType(
    {
        "status": "وضعیت برازش نمایی",
        "family": "خانواده",
        "parameterization": "پارامتردهی",
        "location": "مکان ثابت",
        "rate": "نرخ",
        "mean": "میانگین محاسبه‌شده",
        "observation_count": "تعداد مشاهده‌ها",
        "event_count": "تعداد رویدادهای مشاهده‌شده",
        "censored_count": "تعداد سانسورشده از راست",
        "total_time": "کل زمان آزمون",
        "log_likelihood": "لگاریتم درست‌نمایی",
        "inference": "قابلیت استنباط",
        "censoring_assumption": "فرض سانسورشدگی",
        "failure_code": "کد شکست",
    }
)
_DE_CATALOG: Final[Mapping[str, str]] = MappingProxyType(
    {
        "status": "Status der Exponentialanpassung",
        "family": "Familie",
        "parameterization": "Parametrisierung",
        "location": "Fester Lageparameter",
        "rate": "Rate",
        "mean": "Abgeleiteter Mittelwert",
        "observation_count": "Anzahl der Beobachtungen",
        "event_count": "Anzahl beobachteter Ereignisse",
        "censored_count": "Anzahl rechtszensierter Beobachtungen",
        "total_time": "Gesamte Testzeit",
        "log_likelihood": "Log-Likelihood",
        "inference": "Inferenzumfang",
        "censoring_assumption": "Annahme zur Zensierung",
        "failure_code": "Fehlercode",
    }
)
REPORT_CATALOGS: Final[Mapping[ReportLocale, Mapping[str, str]]] = MappingProxyType(
    {ReportLocale.EN: _EN_CATALOG, ReportLocale.FA: _FA_CATALOG, ReportLocale.DE: _DE_CATALOG}
)

REPORT_TITLES: Final[Mapping[ReportLocale, str]] = MappingProxyType(
    {
        ReportLocale.EN: "Veridist exponential fit report",
        ReportLocale.FA: "گزارش برازش نمایی وریدیست",
        ReportLocale.DE: "Veridist-Bericht zur Anpassung der Exponentialverteilung",
    }
)
REPORT_HEADINGS: Final[Mapping[ReportLocale, str]] = MappingProxyType(
    {
        ReportLocale.EN: "Exponential fit report",
        ReportLocale.FA: "گزارش برازش نمایی",
        ReportLocale.DE: "Bericht zur Anpassung der Exponentialverteilung",
    }
)

FAILURE_MESSAGE_CODES: Final[Mapping[ExponentialFitFailureCode, str]] = MappingProxyType(
    {
        ExponentialFitFailureCode.EMPTY_SAMPLE: "failure.EMPTY_SAMPLE",
        ExponentialFitFailureCode.NO_OBSERVED_EVENTS: "failure.NO_OBSERVED_EVENTS",
        ExponentialFitFailureCode.UNBOUNDED_LIKELIHOOD: "failure.UNBOUNDED_LIKELIHOOD",
        ExponentialFitFailureCode.NUMERICAL_OVERFLOW: "failure.NUMERICAL_OVERFLOW",
    }
)
_FAILURE_MESSAGES: Final[Mapping[ReportLocale, Mapping[str, str]]] = MappingProxyType(
    {
        ReportLocale.EN: MappingProxyType(
            {
                "failure.NONE": "A finite point estimate is available.",
                "failure.EMPTY_SAMPLE": "No observations were supplied.",
                "failure.NO_OBSERVED_EVENTS": (
                    "No event was observed; no finite rate estimate exists."
                ),
                "failure.UNBOUNDED_LIKELIHOOD": (
                    "The likelihood is unbounded; no finite rate estimate exists."
                ),
                "failure.NUMERICAL_OVERFLOW": "Numerical overflow prevented a finite estimate.",
            }
        ),
        ReportLocale.FA: MappingProxyType(
            {
                "failure.NONE": "برآورد نقطه‌ای متناهی در دسترس است.",
                "failure.EMPTY_SAMPLE": "هیچ مشاهده‌ای ارائه نشده است.",
                "failure.NO_OBSERVED_EVENTS": "رویدادی مشاهده نشد؛ برآورد نرخ متناهی وجود ندارد.",
                "failure.UNBOUNDED_LIKELIHOOD": (
                    "درست‌نمایی نامتناهی است؛ برآورد نرخ متناهی وجود ندارد."
                ),
                "failure.NUMERICAL_OVERFLOW": "سرریز عددی مانع برآورد متناهی شد.",
            }
        ),
        ReportLocale.DE: MappingProxyType(
            {
                "failure.NONE": "Eine endliche Punktschätzung ist verfügbar.",
                "failure.EMPTY_SAMPLE": "Es wurden keine Beobachtungen angegeben.",
                "failure.NO_OBSERVED_EVENTS": (
                    "Kein Ereignis wurde beobachtet; es gibt keine endliche Ratenschätzung."
                ),
                "failure.UNBOUNDED_LIKELIHOOD": (
                    "Die Likelihood ist unbeschränkt; es gibt keine endliche Ratenschätzung."
                ),
                "failure.NUMERICAL_OVERFLOW": (
                    "Numerischer Überlauf verhinderte eine endliche Schätzung."
                ),
            }
        ),
    }
)


def _machine_number(value: float) -> str:
    if not isfinite(value):
        raise ValueError("report facts must be finite")
    return repr(value)


def _validate_result(result: ExponentialFit) -> None:
    """Revalidate exact public result types before crossing the HTML boundary."""

    if type(result) is ExponentialFitSuccess:
        ExponentialFitSuccess(
            result.rate,
            result.observation_count,
            result.event_count,
            result.total_time,
            result.mean,
            result.log_likelihood,
            result.censored_count,
            result.provenance,
            result.family,
            result.parameterization,
            result.location,
            result.inference,
            result.censoring_assumption,
        )
        return
    if type(result) is ExponentialFitFailure:
        ExponentialFitFailure(
            result.code, result.observation_count, result.event_count, result.total_time
        )
        return
    raise TypeError("result must be an exact exponential fit result")


def _machine_facts(result: ExponentialFit) -> tuple[dict[str, str], str]:
    if type(result) is ExponentialFitSuccess:
        return (
            {
                "status": "success",
                "family": result.family,
                "parameterization": result.parameterization,
                "location": _machine_number(result.location),
                "rate": _machine_number(result.rate),
                "mean": _machine_number(result.mean),
                "observation_count": str(result.observation_count),
                "event_count": str(result.event_count),
                "censored_count": str(result.censored_count),
                "total_time": _machine_number(result.total_time),
                "log_likelihood": _machine_number(result.log_likelihood),
                "inference": result.inference,
                "censoring_assumption": result.censoring_assumption,
                "failure_code": "none",
            },
            "failure.NONE",
        )
    assert type(result) is ExponentialFitFailure
    total_time = "unavailable" if result.total_time is None else _machine_number(result.total_time)
    return (
        {
            "status": "failure",
            "family": "exponential",
            "parameterization": "rate",
            "location": "0.0",
            "rate": "unavailable",
            "mean": "unavailable",
            "observation_count": str(result.observation_count),
            "event_count": str(result.event_count),
            "censored_count": str(result.observation_count - result.event_count),
            "total_time": total_time,
            "log_likelihood": "unavailable",
            "inference": "not_provided",
            "censoring_assumption": "independent_right_censoring",
            "failure_code": result.code.value,
        },
        FAILURE_MESSAGE_CODES[result.code],
    )


def _latin(value: str) -> str:
    return f'<bdi dir="ltr" class="latin">{escape(value, quote=True)}</bdi>'


def render_exponential_report(result: ExponentialFit, locale: ReportLocale) -> str:
    """Render a pure escaped HTML report for one explicitly selected locale."""

    if type(locale) is not ReportLocale:
        raise TypeError("locale must be an exact ReportLocale")
    _validate_result(result)
    facts, failure_message_code = _machine_facts(result)
    if tuple(facts) != REPORT_KEYS:
        raise ValueError("report facts do not satisfy the fixed report schema")
    direction = "rtl" if locale is ReportLocale.FA else "ltr"
    rows = "".join(
        f'<div class="fact" data-report-key="{key}" data-label-key="{REPORT_LABEL_KEYS[key]}" '
        f'data-machine-value="{escape(facts[key], quote=True)}">'
        f'<span class="label">{escape(REPORT_CATALOGS[locale][key])}</span>'
        f'<span class="machine-id">{_latin(key)}</span>{_latin(facts[key])}</div>'
        for key in REPORT_KEYS
    )
    failure_message = _FAILURE_MESSAGES[locale][failure_message_code]
    return (
        f'<!doctype html><html lang="{locale.value}" dir="{direction}"><head>'
        '<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(REPORT_TITLES[locale])}</title>"
        "<style>.report{max-width:58rem;margin:1rem auto;"
        "font-family:system-ui,sans-serif;text-align:start}"
        '[dir="rtl"] .report{text-align:right}'
        ".fact{display:grid;grid-template-columns:minmax(12rem,1fr) auto auto;"
        "gap:.6rem;padding:.35rem 0}"
        ".latin{direction:ltr;unicode-bidi:isolate;display:inline-block}.machine-id{opacity:.72;font-size:.85em}"
        '</style></head><body><main class="report" dir="'
        f'{direction}"><h1>{escape(REPORT_HEADINGS[locale])}</h1>'
        f'<p class="formula">{_latin("r*log(rate)-rate*tau")}</p>{rows}'
        f'<div class="failure" data-failure-message-code="{failure_message_code}">'
        f"{escape(failure_message)}</div>"
        "</main></body></html>"
    )


__all__ = [
    "FAILURE_MESSAGE_CODES",
    "REPORT_CATALOGS",
    "REPORT_KEYS",
    "REPORT_LABEL_KEYS",
    "REPORT_HEADINGS",
    "REPORT_TITLES",
    "ReportLocale",
    "render_exponential_report",
]
