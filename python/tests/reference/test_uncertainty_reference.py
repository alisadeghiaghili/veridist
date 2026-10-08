"""Uncertainty against independent references: covariance, intervals, derived quantities.

Provenance of every constant below: scipy 1.18.1 with numpy 2.5.3 (``scipy.stats`` log-density
and log-survival for the likelihood; nothing here imports scipy).  The datasets are frozen
literals (drawn once with ``numpy.random.default_rng(seed)`` and rounded to three decimals; the
seed is given at each dataset).  The right-censored variant of each dataset censors every value
above its censoring point at that point.

* Estimates: Nelder-Mead on the negative log-likelihood, restarted until the simplex stopped
  moving (``xatol=1e-13``).
* Covariance: the inverse of the negative Hessian at the estimate, the Hessian taken with
  five-point central differences of the scipy log-likelihood (step 0.4% of each parameter).
* Wald intervals: log scale for positive parameters, identity otherwise, level 0.95.
* Profile intervals: the nuisance parameter maximized by ``scipy.optimize.minimize_scalar``
  (bounded) polished by Nelder-Mead, the crossing of ``2 (l_max - l_profile) = chi2_1(0.95)``
  found by ``scipy.optimize.brentq`` with ``xtol=1e-13``; an independent implementation of the
  profile that shares no code with the package.
* Derived quantities: the mean, the 0.1 quantile (B10) and the survival at the sample median;
  gradients by five-point central differences of the scipy functions, delta method on the log
  scale (mean, quantile) or the logit scale (survival); profile intervals for the exponential
  and Weibull families by reparameterizing the scale through the quantity.
* ``exact``: ``scipy.stats.chi2.ppf`` with ``2 r`` degrees of freedom.

The acceptance tolerance is a relative 1e-6: the references themselves are limited by their
finite differences (about 1e-7) and the package fits by the golden-section search (about 1e-8).
"""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable
from typing import Any

import mpmath as mp

from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.domain.values import ExactValue, RightCensoredValue
from veridist.families import FitSuccess, FitUncertainty, fit
from veridist.families._models import ExponentialModel, Model, build_model, normal_hazard
from veridist.families._reliability import expand_bracket
from veridist.families.registry import FamilyId
from veridist.statistics._special import digamma, trigamma

DATASETS = {
    "exponential": (  # numpy.random.default_rng(101), rounded to three decimals
        32.718,
        4.625,
        7.651,
        8.262,
        5.12,
        3.047,
        21.803,
        4.989,
        32.57,
        6.638,
        3.297,
        0.353,
        49.687,
        31.853,
        2.041,
        9.326,
        12.395,
        12.438,
        9.578,
        5.107,
        0.42,
        2.662,
        34.004,
        6.567,
        9.889,
        11.912,
        15.347,
        6.744,
        1.588,
        11.921,
    ),
    "weibull_min": (  # numpy.random.default_rng(102), rounded to three decimals
        6.41,
        18.079,
        11.543,
        8.066,
        11.76,
        8.076,
        12.999,
        9.848,
        30.266,
        2.411,
        6.046,
        16.552,
        14.131,
        17.757,
        4.22,
        3.993,
        19.042,
        24.645,
        6.077,
        7.946,
        2.689,
        7.367,
        12.154,
        5.318,
        3.352,
        18.322,
        13.627,
        12.783,
        13.353,
        12.969,
    ),
    "lognormal": (  # numpy.random.default_rng(103), rounded to three decimals
        14.325,
        3.429,
        10.901,
        3.598,
        14.056,
        2.905,
        4.824,
        5.135,
        2.089,
        12.325,
        2.913,
        3.372,
        5.266,
        11.548,
        4.536,
        24.368,
        7.279,
        6.213,
        9.671,
        6.83,
        3.917,
        5.313,
        8.167,
        19.033,
        8.654,
        9.829,
        1.93,
        6.292,
        7.266,
        21.239,
    ),
    "gamma": (  # numpy.random.default_rng(104), rounded to three decimals
        12.415,
        5.494,
        8.67,
        28.917,
        7.059,
        16.31,
        8.157,
        13.967,
        24.346,
        15.557,
        23.202,
        5.495,
        19.707,
        15.221,
        2.75,
        17.429,
        8.46,
        14.673,
        13.227,
        1.944,
        8.374,
        11.727,
        2.416,
        8.636,
        2.954,
        20.488,
        9.419,
        7.146,
        6.307,
        10.946,
    ),
    "normal": (  # numpy.random.default_rng(105), rounded to three decimals
        49.492,
        59.994,
        65.904,
        49.573,
        48.087,
        58.496,
        50.468,
        56.507,
        62.811,
        54.481,
        58.706,
        31.356,
        58.399,
        48.733,
        42.52,
        48.839,
        40.527,
        40.637,
        33.685,
        44.898,
        39.086,
        43.604,
        56.475,
        50.468,
        39.702,
        53.634,
        57.972,
        50.201,
        50.104,
        49.084,
    ),
    "gumbel_right": (  # numpy.random.default_rng(106), rounded to three decimals
        41.622,
        69.13,
        62.195,
        52.829,
        42.77,
        61.406,
        55.589,
        59.284,
        55.679,
        67.206,
        49.938,
        65.867,
        42.162,
        60.848,
        56.302,
        67.604,
        48.668,
        54.044,
        51.879,
        49.037,
        68.643,
        57.19,
        53.665,
        63.964,
        64.72,
        82.403,
        48.929,
        53.536,
        45.355,
        61.309,
    ),
}
CENSORING_POINTS = {
    "exponential": 12.427,
    "weibull_min": 14.005,
    "lognormal": 10.633,
    "gamma": 15.473,
    "normal": 56.499,
    "gumbel_right": 63.522,
}

# case -> params, cov, se, wald, profile, derived (kind -> (argument, wald, profile)), exact
REFERENCE = {
    "exponential/uncensored": {
        "params": (0.0822927891364,),
        "cov": ((0.000225736771512,),),
        "se": (0.0150245389783,),
        "wald": ((0.0575379458659, 0.117698034609),),
        "profile": ((0.0562478855721, 0.115352699189),),
        "derived": {
            "mean": (
                None,
                (12.1517329829, 8.49631859776, 17.3798349),
                (12.1517329829, 8.66906459087, 17.7784460665),
            ),
            "quantile": (
                0.1,
                (1.28031285321, 0.895176508653, 1.83114836711),
                (1.28031285321, 0.913377115565, 1.87314624516),
            ),
            "survival": (
                7.9565,
                (0.519565404844, 0.3990606147, 0.637835978009),
                (0.519565404844, 0.399395249926, 0.639201011839),
            ),
        },
        "exact": (0.055522597658, 0.114246629942),
    },
    "exponential/censored": {
        "params": (0.0941990514902,),
        "cov": ((0.000403339150168,),),
        "se": (0.0200833052601,),
        "wald": ((0.0620254253233, 0.143061676005),),
        "profile": ((0.0601174494278, 0.13922494655),),
        "derived": {
            "mean": (
                None,
                (10.6158181444, 6.98999220724, 16.1224206744),
                (10.6158181444, 7.18262082178, 16.6341055637),
            ),
            "quantile": (
                0.1,
                (1.11848807383, 0.736469183399, 1.69866655591),
                (1.11848807383, 0.756764633558, 1.75257793969),
            ),
            "survival": (
                7.9565,
                (0.47260527474, 0.331030288355, 0.618727690585),
                (0.47260527474, 0.330303520806, 0.619821040867),
            ),
        },
    },
    "weibull_min/uncensored": {
        "params": (1.84531768247, 12.8680603675),
        "cov": ((0.0672224688093, 0.112003436334), (0.112003436334, 1.80753971084)),
        "se": (0.259272961971, 1.34444773451),
        "wald": ((1.40111776105, 2.43034343285), (10.4852799299, 15.7923277898)),
        "profile": ((1.37143288132, 2.3865716679), (10.3581059056, 15.8182639776)),
        "derived": {
            "mean": (
                None,
                (11.4307431243, 9.34504406272, 13.9819446005),
                (11.4307431243, 9.27993381824, 14.0486460737),
            ),
            "quantile": (
                0.1,
                (3.80093463368, 2.43332229918, 5.93719298688),
                (3.80093463368, 2.22032264877, 5.57102682697),
            ),
            "survival": (
                11.6515,
                (0.434939764057, 0.300141000329, 0.5800983013),
                (0.434939764057, 0.298209692308, 0.579508855913),
            ),
        },
    },
    "weibull_min/censored": {
        "params": (2.00571809959, 12.427913767),
        "cov": ((0.141152425848, -0.0243640815688), (-0.0243640815688, 1.74936041893)),
        "se": (0.375702576312, 1.32263389452),
        "wald": ((1.38939727888, 2.89543182223), (10.0881045528, 15.3104123566)),
        "profile": ((1.35094840329, 2.82933134098), (10.1316599583, 15.8561583898)),
        "derived": {
            "mean": (
                None,
                (11.0133894, 8.93618967214, 13.5734301224),
                (11.0133894, 9.00479242144, 14.1409041809),
            ),
            "quantile": (
                0.1,
                (4.0469729186, 2.57402091189, 6.36280370846),
                (4.0469729186, 2.26827445185, 5.85984839892),
            ),
            "survival": (
                11.6515,
                (0.415350723154, 0.274645143252, 0.57136032224),
                (0.415350723154, 0.272598276086, 0.571053700371),
            ),
        },
    },
    "lognormal/uncensored": {
        "params": (1.89639444764, 0.654854236723),
        "cov": ((0.0142944690447, 6.47878177853e-11), (6.47878177853e-11, 0.00714723466538)),
        "se": (0.119559479108, 0.0845413192787),
        "wald": ((1.66206217458, 2.13072672071), (0.508457893429, 0.843401345315)),
        "profile": ((1.65435667769, 2.13843221165), (0.518552382694, 0.86325407765)),
        "derived": {
            "mean": (None, (8.25493731135, 6.37622309943, 10.6872029024), None),
            "quantile": (0.1, (2.87819964487, 2.09788640432, 3.94875202903), None),
            "survival": (6.561, (0.50929044182, 0.369588662875, 0.647556094696), None),
        },
    },
    "lognormal/censored": {
        "params": (1.90512398628, 0.6721442097),
        "cov": ((0.0166389628716, 0.00253423388293), (0.00253423388293, 0.0117212145975)),
        "se": (0.128992103912, 0.108264558363),
        "wald": ((1.65230410832, 2.15794386424), (0.490181062026, 0.9216550243)),
        "profile": ((1.65475862002, 2.18767486461), (0.504367779798, 0.953819343958)),
        "derived": {
            "mean": (None, (8.42339459531, 6.16574207273, 11.5077107786), None),
            "quantile": (0.1, (2.83980822097, 2.02933169397, 3.97397367609), None),
            "survival": (6.561, (0.51423055913, 0.368095068241, 0.657972713649), None),
        },
    },
    "gamma/uncensored": {
        "params": (2.6566417556, 4.40923834401),
        "cov": ((0.419272585328, -0.695868295967), (-0.695868295967, 1.39886956197)),
        "se": (0.64751261403, 1.18273816289),
        "wald": ((1.64765642568, 4.28350553406), (2.60637251933, 7.4591727123)),
        "profile": ((1.59014109041, 4.14693131964), (2.72516978657, 7.88676699162)),
        "derived": {
            "mean": (None, (11.7137666951, 9.40480845647, 14.5895932727), None),
            "quantile": (0.1, (3.95057029835, 2.60638743691, 5.98798377445), None),
            "survival": (10.1825, (0.50594882029, 0.367989566393, 0.64300807623), None),
        },
    },
    "gamma/censored": {
        "params": (2.43151946646, 4.99149881881),
        "cov": ((0.481272971931, -1.10553060327), (-1.10553060327, 2.95751887401)),
        "se": (0.693738403096, 1.71974383965),
        "wald": ((1.39001947167, 4.2533842412), (2.54075404201, 9.80616779359)),
        "profile": ((1.33301093948, 4.08680670335), (2.72474807543, 10.7597245259)),
        "derived": {
            "mean": (None, (12.1369265447, 9.33535351219, 15.7792616809), None),
            "quantile": (0.1, (3.82425087986, 2.42308542284, 6.03564969451), None),
            "survival": (10.1825, (0.519025035989, 0.373477838803, 0.661413425894), None),
        },
    },
    "normal/uncensored": {
        "params": (49.8147666148, 8.29287010085),
        "cov": ((2.29238981698, -1.43274182282e-08), (-1.43274182282e-08, 1.14619491558)),
        "se": (1.5140640069, 1.07060492974),
        "wald": ((46.847255691, 52.7822775387), (6.43895241546, 10.6805719428)),
        "profile": ((46.7496757537, 52.8798575796), (6.56678590254, 10.9319808309)),
        "derived": {
            "mean": (None, (49.8147666148, 46.847255691, 52.7822775387), None),
            "quantile": (0.1, (39.1870259542, 35.1823293053, 43.1917226032), None),
            "survival": (49.8385, (0.498858267674, 0.359946379617, 0.637946632183), None),
        },
    },
    "normal/censored": {
        "params": (50.3902913175, 9.1941555585),
        "cov": ((3.11599345717, 0.474500310689), (0.474500310689, 2.18782813317)),
        "se": (1.76521767983, 1.47913087087),
        "wald": ((46.9305282402, 53.8500543948), (6.70768788283, 12.6023300294)),
        "profile": ((46.9647245192, 54.2564235465), (6.90150109249, 13.0415234269)),
        "derived": {
            "mean": (None, (50.3902913175, 46.9305282402, 53.8500543948), None),
            "quantile": (0.1, (38.6075068676, 34.0138960063, 43.201117729), None),
            "survival": (49.8385, (0.523928332378, 0.377482486004, 0.666371803412), None),
        },
    },
    "gumbel_right/uncensored": {
        "params": (52.6811666642, 8.13843399643),
        "cov": ((2.47163317882, 0.581282193429), (0.581282193429, 1.28070935909)),
        "se": (1.57214286209, 1.13168430187),
        "wald": ((49.5998232759, 55.7625100524), (6.19694447262, 10.6881880589)),
        "profile": ((49.5797711828, 55.9430253155), (6.30470374667, 10.9137012884)),
        "derived": {
            "mean": (None, (57.3787982547, 53.6758729742, 61.0817235352), None),
            "quantile": (0.1, (45.8934486576, 42.8615857288, 48.9253115865), None),
            "survival": (55.9905, (0.486184160756, 0.344961187055, 0.629648162368), None),
        },
    },
    "gumbel_right/censored": {
        "params": (53.1484279046, 8.95753342276),
        "cov": ((3.10189323172, 0.943451299547), (0.943451299547, 2.11191629887)),
        "se": (1.76121924578, 1.45324337221),
        "wald": ((49.696501614, 56.6003541952), (6.51766523887, 12.3107588499)),
        "profile": ((49.7315088721, 56.9498934548), (6.68642496784, 12.7093667799)),
        "derived": {
            "mean": (None, (58.3188565151, 53.982644112, 62.6550689181), None),
            "quantile": (0.1, (45.6775544006, 42.2843673014, 49.0707414998), None),
            "survival": (55.9905, (0.517186460512, 0.36804802982, 0.663324738182), None),
        },
    },
}

LIFETIME_FAMILIES = ("exponential", "weibull_min", "lognormal", "gamma")
SETTINGS = ("uncensored", "censored")
REL = 1e-6
CHI_SQUARE_1_95 = 3.841458820694124  # scipy.stats.chi2.ppf(0.95, 1)


def _sample(family: str, setting: str) -> tuple[list[float], list[float]]:
    data = DATASETS[family]
    if setting == "uncensored":
        return list(data), []
    cut = CENSORING_POINTS[family]
    return [x for x in data if x <= cut], [cut] * sum(x > cut for x in data)


def _observations(family: str, setting: str) -> list[object]:
    exact, censored = _sample(family, setting)
    if family in LIFETIME_FAMILIES:
        return [ExactLifetime(x) for x in exact] + [RightCensoredLifetime(x) for x in censored]
    return [ExactValue(x) for x in exact] + [RightCensoredValue(x) for x in censored]


def _model(family: str, setting: str) -> Model:
    exact, censored = _sample(family, setting)
    if family == "exponential":
        return ExponentialModel(len(exact), math.fsum(exact + censored), len(censored))
    return build_model(FamilyId(family), tuple(exact), tuple(censored))


def _cases() -> list[tuple[str, str]]:
    return [(family, setting) for family in DATASETS for setting in SETTINGS]


def central_difference_hessian(
    function: Callable[[tuple[float, ...]], float], theta: tuple[float, ...], rel: float = 0.002
) -> list[list[float]]:
    """Five-point (Richardson-extrapolated) central-difference Hessian, step ``rel * |theta|``."""

    size = len(theta)
    steps = [rel * max(abs(value), 1e-2) for value in theta]

    def at(*moves: tuple[int, float]) -> float:
        shifted = list(theta)
        for index, move in moves:
            shifted[index] += move
        return function(tuple(shifted))

    hessian = [[0.0] * size for _ in range(size)]
    for i in range(size):
        h = steps[i]
        hessian[i][i] = (
            -at((i, 2 * h)) + 16 * at((i, h)) - 30 * at() + 16 * at((i, -h)) - at((i, -2 * h))
        ) / (12 * h * h)
    for i in range(size):
        for j in range(i + 1, size):
            h, k = steps[i], steps[j]

            def slope(offset: float, i: int = i, j: int = j, k: float = k) -> float:
                return (
                    8 * (at((i, offset), (j, k)) - at((i, offset), (j, -k)))
                    - (at((i, offset), (j, 2 * k)) - at((i, offset), (j, -2 * k)))
                ) / (12 * k)

            hessian[i][j] = hessian[j][i] = (
                8 * (slope(h) - slope(-h)) - (slope(2 * h) - slope(-2 * h))
            ) / (12 * h)
    return hessian


class UncertaintyReferenceTests(unittest.TestCase):
    def assertRelative(
        self, actual: float, expected: float, rel: float, scale: float = 0.0
    ) -> None:
        bound = rel * max(abs(expected), scale)
        self.assertLessEqual(abs(actual - expected), bound, (actual, expected))

    def _uncertainty(self, family: str, setting: str) -> tuple[Any, FitUncertainty]:
        result = fit(family, _observations(family, setting))
        self.assertIsInstance(result, FitSuccess)
        uncertainty = result.uncertainty()  # type: ignore[union-attr]
        self.assertIsInstance(uncertainty, FitUncertainty)
        return result, uncertainty

    def test_estimates_covariance_and_standard_errors(self) -> None:
        for family, setting in _cases():
            with self.subTest(family=family, setting=setting):
                case = REFERENCE[f"{family}/{setting}"]
                result, uncertainty = self._uncertainty(family, setting)
                names = tuple(result.parameters)
                self.assertEqual(uncertainty.parameter_names, names)
                for index, name in enumerate(names):
                    self.assertRelative(result.parameters[name], case["params"][index], REL)
                    self.assertRelative(uncertainty.standard_errors[name], case["se"][index], REL)
                    for other in range(len(names)):
                        scale = math.sqrt(case["cov"][index][index] * case["cov"][other][other])
                        self.assertRelative(
                            uncertainty.covariance[index][other],
                            case["cov"][index][other],
                            REL,
                            scale,
                        )

    def test_wald_and_profile_intervals(self) -> None:
        for family, setting in _cases():
            with self.subTest(family=family, setting=setting):
                case = REFERENCE[f"{family}/{setting}"]
                result, uncertainty = self._uncertainty(family, setting)
                wald = uncertainty.confidence_intervals(0.95, "wald")
                profile = uncertainty.confidence_intervals(0.95, "profile")
                for index, name in enumerate(result.parameters):
                    for side in (0, 1):
                        self.assertRelative(wald[name][side], case["wald"][index][side], REL)
                        self.assertRelative(profile[name][side], case["profile"][index][side], REL)
                details = uncertainty.interval_details(0.95, "profile")
                for interval in details.values():
                    self.assertFalse(interval.lower_unbounded or interval.upper_unbounded)

    def test_derived_quantities_against_delta_method_and_profile(self) -> None:
        for family, setting in _cases():
            with self.subTest(family=family, setting=setting):
                case = REFERENCE[f"{family}/{setting}"]
                _, uncertainty = self._uncertainty(family, setting)
                for kind, (argument, wald, profile) in case["derived"].items():
                    call = {
                        "mean": lambda **kw: uncertainty.mean(**kw),
                        "quantile": lambda **kw: uncertainty.quantile(argument, **kw),
                        "survival": lambda **kw: uncertainty.survival(argument, **kw),
                    }[kind]
                    estimate = call(method="wald")
                    self.assertEqual(estimate.method, "wald")
                    self.assertEqual(estimate.level, 0.95)
                    for got, expected in zip(
                        (estimate.estimate, estimate.lower, estimate.upper), wald, strict=True
                    ):
                        self.assertRelative(got, expected, REL)
                    if profile is None:
                        continue
                    estimate = call(method="profile")
                    self.assertEqual(estimate.method, "profile")
                    for got, expected in zip(
                        (estimate.estimate, estimate.lower, estimate.upper), profile, strict=True
                    ):
                        self.assertRelative(got, expected, REL)

    def test_exact_exponential_interval(self) -> None:
        _, uncertainty = self._uncertainty("exponential", "uncensored")
        expected = REFERENCE["exponential/uncensored"]["exact"]
        interval = uncertainty.confidence_intervals(0.95, "exact")["rate"]
        self.assertRelative(interval[0], expected[0], 1e-12)
        self.assertRelative(interval[1], expected[1], 1e-12)

    def test_profile_endpoints_solve_the_likelihood_ratio_equation(self) -> None:
        """Re-derive the deviance at every endpoint with an independent nested search."""

        for family, setting in _cases():
            with self.subTest(family=family, setting=setting):
                result, uncertainty = self._uncertainty(family, setting)
                model = _model(family, setting)
                theta = tuple(result.parameters.values())
                intervals = uncertainty.confidence_intervals(0.95, "profile")
                for index, name in enumerate(result.parameters):
                    for endpoint in intervals[name]:
                        pll = self._profile(model, theta, index, endpoint)
                        deviance = 2.0 * (result.log_likelihood - pll)
                        self.assertAlmostEqual(deviance, CHI_SQUARE_1_95, delta=1e-7)

    @staticmethod
    def _profile(model: Model, theta: tuple[float, ...], index: int, value: float) -> float:
        if len(theta) == 1:
            return model.log_likelihood((value,))
        other = 1 - index
        positive = model.positive[other]
        center = math.log(theta[other]) if positive else theta[other]
        half = 3.0 if positive else 5.0 * abs(theta[other]) + 1.0

        def objective(coordinate: float) -> float:
            built = [0.0, 0.0]
            built[index] = value
            built[other] = math.exp(coordinate) if positive else coordinate
            try:
                return model.log_likelihood(tuple(built))
            except (ArithmeticError, ValueError):
                return -math.inf

        _, best, _ = expand_bracket(
            objective,
            lower=center - half,
            upper=center + half,
            hard_lower=center - 20 * half,
            hard_upper=center + 20 * half,
            steps=120,
        )
        return best

    def test_model_log_likelihood_equals_the_fit_log_likelihood(self) -> None:
        for family, setting in _cases():
            with self.subTest(family=family, setting=setting):
                result, _ = self._uncertainty(family, setting)
                model = _model(family, setting)
                value = model.log_likelihood(tuple(result.parameters.values()))
                self.assertRelative(value, result.log_likelihood, 1e-12)


class AnalyticHessianTests(unittest.TestCase):
    def test_hessian_matches_central_differences_at_and_away_from_the_estimate(self) -> None:
        for family, setting in _cases():
            if family == "exponential":
                continue
            case = REFERENCE[f"{family}/{setting}"]
            model = _model(family, setting)
            estimate = case["params"]
            away = tuple(
                value * (1.15 if index == 0 else 0.9) if model.positive[index] else value + 1.5
                for index, value in enumerate(estimate)
            )
            for label, theta in (("estimate", estimate), ("away", away)):
                with self.subTest(family=family, setting=setting, at=label):
                    evaluation = model.evaluate(theta)
                    reference = central_difference_hessian(model.log_likelihood, theta)
                    for i in range(2):
                        for j in range(2):
                            scale = math.sqrt(abs(reference[i][i] * reference[j][j]))
                            self.assertLessEqual(
                                abs(evaluation.hessian[i][j] - reference[i][j]), 1e-6 * scale
                            )
                    self.assertEqual(evaluation.hessian[0][1], evaluation.hessian[1][0])

    def test_gradient_matches_central_differences_away_from_the_estimate(self) -> None:
        for family, setting in _cases():
            case = REFERENCE[f"{family}/{setting}"]
            model = _model(family, setting)
            away = tuple(
                value * (1.15 if index == 0 else 0.9) if model.positive[index] else value + 1.5
                for index, value in enumerate(case["params"])
            )
            with self.subTest(family=family, setting=setting):
                evaluation = model.evaluate(away)
                for index, value in enumerate(away):
                    h = 1e-3 * max(abs(value), 1e-2)

                    def at(offset: float, index: int = index) -> float:
                        shifted = list(away)
                        shifted[index] += offset
                        return model.log_likelihood(tuple(shifted))

                    slope = (8 * (at(h) - at(-h)) - (at(2 * h) - at(-2 * h))) / (12 * h)
                    self.assertLessEqual(abs(evaluation.gradient[index] - slope), 1e-6 * abs(slope))

    def test_gradient_is_zero_at_the_fitted_estimate(self) -> None:
        for family, setting in _cases():
            case = REFERENCE[f"{family}/{setting}"]
            model = _model(family, setting)
            with self.subTest(family=family, setting=setting):
                evaluation = model.evaluate(case["params"])
                for index, slope in enumerate(evaluation.gradient):
                    # The Newton decrement g^2 / (2 |h|): what the fit could still gain.
                    decrement = slope * slope / (2.0 * abs(evaluation.hessian[index][index]))
                    self.assertLess(decrement, 1e-9)

    def test_gamma_censored_derivatives_against_mpmath(self) -> None:
        mp.mp.dps = 30
        exact, censored = _sample("gamma", "censored")
        exact_values = [mp.mpf(x) for x in exact]
        censored_values = [mp.mpf(x) for x in censored]

        def log_likelihood(shape: Any, scale: Any) -> Any:
            total = mp.mpf(0)
            for x in exact_values:
                total += -mp.loggamma(shape) - shape * mp.log(scale) + (shape - 1) * mp.log(x)
                total -= x / scale
            for x in censored_values:
                total += mp.log(mp.gammainc(shape, x / scale, mp.inf, regularized=True))
            return total

        model = _model("gamma", "censored")
        for theta in (tuple(REFERENCE["gamma/censored"]["params"]), (3.1, 3.9)):
            point = (mp.mpf(theta[0]), mp.mpf(theta[1]))
            gradient = (
                mp.diff(log_likelihood, point, (1, 0)),
                mp.diff(log_likelihood, point, (0, 1)),
            )
            hessian = (
                (mp.diff(log_likelihood, point, (2, 0)), mp.diff(log_likelihood, point, (1, 1))),
                (mp.diff(log_likelihood, point, (1, 1)), mp.diff(log_likelihood, point, (0, 2))),
            )
            for precise, rel in ((True, 1e-6), (False, 1e-5)):
                with self.subTest(theta=theta, precise=precise):
                    evaluation = model.evaluate(theta, precise=precise)
                    self.assertAlmostEqual(
                        evaluation.value, float(log_likelihood(*point)), delta=1e-11
                    )
                    for i in range(2):
                        scale = math.sqrt(abs(float(hessian[i][i])))
                        self.assertLessEqual(
                            abs(evaluation.gradient[i] - float(gradient[i])),
                            rel * max(abs(float(gradient[i])), scale),
                        )
                        for j in range(2):
                            self.assertLessEqual(
                                abs(evaluation.hessian[i][j] - float(hessian[i][j])),
                                rel * math.sqrt(abs(float(hessian[0][0] * hessian[1][1]))),
                            )


class SpecialFunctionTests(unittest.TestCase):
    POINTS = (
        1e-8, 1e-3, 0.01, 0.1, 0.5, 1.0, 2.0, 3.5, 7.5, 10.0, 19.9, 20.0, 25.0, 100.0, 1e4, 1e8,
    )  # fmt: skip

    def test_trigamma_matches_mpmath(self) -> None:
        mp.mp.dps = 40
        for x in self.POINTS:
            expected = float(mp.polygamma(1, mp.mpf(x)))
            self.assertLessEqual(abs(trigamma(x) - expected), 1e-13 * expected, x)

    def test_digamma_matches_mpmath(self) -> None:
        mp.mp.dps = 40
        for x in self.POINTS:
            expected = float(mp.digamma(mp.mpf(x)))
            self.assertLessEqual(abs(digamma(x) - expected), 1e-13 * max(abs(expected), 1.0), x)

    def test_arguments_must_be_positive(self) -> None:
        for function in (digamma, trigamma):
            for bad in (0.0, -1.0, math.nan):
                with self.assertRaises(ValueError):
                    function(bad)

    def test_normal_hazard_matches_mpmath(self) -> None:
        mp.mp.dps = 50
        for z in (-12.0, -3.0, -0.5, 0.0, 0.7, 4.0, 11.9, 12.0, 13.0, 30.0, 1e3, 1e8):
            density = mp.exp(-(mp.mpf(z) ** 2) / 2) / mp.sqrt(2 * mp.pi)
            tail = mp.erfc(mp.mpf(z) / mp.sqrt(2)) / 2
            hazard = float(density / tail)
            excess = float(density / tail - mp.mpf(z))
            got, got_excess = normal_hazard(z)
            self.assertLessEqual(abs(got - hazard), 1e-12 * hazard, z)
            self.assertLessEqual(abs(got_excess - excess), 1e-9 * abs(excess) + 1e-12, z)


if __name__ == "__main__":
    unittest.main()
