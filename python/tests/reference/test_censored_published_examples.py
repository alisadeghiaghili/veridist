"""Published examples of censored and truncated fits, recorded as cited constants (CENS-REF-01).

Nothing is recorded here yet.  A published value enters this file only with its source, the
version of the source and the recipe that extracts it, and only where it can be read from an
accessible copy of the source; no value is typed from memory.  The candidates, none verified
yet, are listed so that the work is not forgotten:

* TODO: the interval-censored breast cosmesis data of Finkelstein (1986) with a Weibull fit, as
  worked in the ``survival`` and ``interval`` R packages (source: the package vignette and the
  paper's table of retraction times; the redistribution status of the data is to be confirmed);
* TODO: ``smokedfish`` (left-, right- and interval-censored concentrations) and ``salinity`` of
  ``fitdistrplus::fitdistcens`` (source: the package's data documentation, which lists the
  structure but not the rows or the fit output; the rows and the printed estimates are to be
  taken from the package vignette or the paper of Delignette-Muller and Dutang, 2015);
* TODO: the left-truncated Channing House data of Klein and Moeschberger (2003) (source: the
  textbook's data appendix);
* TODO: the inspection and failure data sets of Meeker and Escobar (1998) and Lawless (2003).

Results of ``survival::survreg``, ``fitdistrplus::fitdistcens``, ``flexsurv::flexsurvreg``,
lifelines and SurPyval are cross-checks of the same kind and are recorded the same way.  Until a
constant is recorded, the agreement with an independent oracle is established by
``test_censored_fits_reference`` (a 30-digit maximisation for every family, censoring kind and
truncation type).
"""

from __future__ import annotations

import unittest

#: Cited constants: ``name -> (source, version, recipe, data, expected parameters, tolerance)``.
PUBLISHED: dict[str, tuple[str, str, str, object, dict[str, float], float]] = {}


class PublishedExampleTests(unittest.TestCase):
    def test_no_example_is_recorded_without_its_source(self) -> None:
        for name, (source, version, recipe, _, expected, tolerance) in PUBLISHED.items():
            with self.subTest(example=name):
                self.assertTrue(source and version and recipe)
                self.assertGreater(tolerance, 0.0)
                self.assertTrue(expected)

    @unittest.skip("TODO: no published example is verified yet; see the module docstring")
    def test_the_published_examples_are_reproduced(self) -> None:
        self.fail("no example")


if __name__ == "__main__":
    unittest.main()
