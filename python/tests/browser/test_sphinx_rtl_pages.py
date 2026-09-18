"""Real-browser contracts for rendered Sphinx locale pages."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PYTHON_ROOT / "docs" / "source"


@unittest.skipUnless(
    os.environ.get("VERIDIST_BROWSER_TESTS") == "1",
    "Sphinx browser evidence runs only in the dedicated Chromium job",
)
class SphinxRtlBrowserContracts(unittest.TestCase):
    def test_rtl_doc01_built_farsi_and_german_pages_have_computed_direction_contracts(self) -> None:
        from playwright.sync_api import sync_playwright

        configured_directory = os.environ.get("VERIDIST_BROWSER_ARTIFACT_DIR")
        with ExitStack() as resources:
            root = Path(resources.enter_context(tempfile.TemporaryDirectory()))
            artifact_directory = (
                Path(configured_directory)
                if configured_directory is not None
                else Path(resources.enter_context(tempfile.TemporaryDirectory()))
            )
            artifact_directory.mkdir(parents=True, exist_ok=True)
            outputs = {locale: root / locale for locale in ("fa", "de")}
            for locale, output in outputs.items():
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "sphinx",
                        "-b",
                        "html",
                        "-W",
                        "-n",
                        str(SOURCE_ROOT),
                        str(output),
                        "-D",
                        f"language={locale}",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            with sync_playwright() as playwright:
                executable = os.environ.get("VERIDIST_BROWSER_EXECUTABLE")
                browser = playwright.chromium.launch(
                    **({} if executable is None else {"executable_path": executable})
                )
                try:
                    page = browser.new_page()
                    page_contracts = {
                        "api.html": ("code", "pre"),
                        "exponential-right-censoring.html": (
                            "code",
                            "pre",
                            "table",
                            "formula",
                        ),
                        "families-log-density-likelihood.html": ("code", "pre", "table", "math"),
                        "index.html": ("code", "pre"),
                    }
                    for page_name, required_exemplars in page_contracts.items():
                        page.goto((outputs["fa"] / page_name).as_uri(), wait_until="load")
                        fa = page.evaluate("""(requiredExemplars) => ({
                      lang: document.documentElement.lang, dir: document.documentElement.dir,
                      body: {
                        direction: getComputedStyle(document.body).direction,
                        align: getComputedStyle(document.body).textAlign,
                      },
                      networkScripts: [...document.scripts]
                        .map((script) => script.src)
                        .filter((source) => /^https?:/i.test(source)),
                      exemplars: Object.fromEntries([
                        ['code', 'code.literal'],
                        ['pre', '.highlight pre'],
                        ['table', 'table.docutils'],
                        ['math', '.math'],
                        ['formula', 'p[dir="ltr"][align="center"]'],
                      ].filter(([name]) => requiredExemplars.includes(name))
                        .map(([name, selector]) => {
                        const element = document.querySelector(selector);
                        if (element === null) {
                          throw new Error(`missing required exemplar: ${selector}`);
                        }
                        const style = getComputedStyle(element);
                        return [name, {direction: style.direction, unicodeBidi: style.unicodeBidi}];
                      })),
                      urlExemplars: [...document.querySelectorAll(
                        'a.veridist-api-url'
                      )].map((element) => {
                        const style = getComputedStyle(element);
                        const box = element.getBoundingClientRect();
                        return {
                          direction: style.direction,
                          unicodeBidi: style.unicodeBidi,
                          visible: box.width > 0 && box.height > 0,
                        };
                      }),
                        })""", list(required_exemplars))
                        self.assertEqual(fa["lang"], "fa", page_name)
                        self.assertEqual(fa["dir"], "rtl", page_name)
                        self.assertEqual(
                            fa["body"], {"direction": "rtl", "align": "right"}, page_name
                        )
                        self.assertEqual(fa["networkScripts"], [], page_name)
                        self.assertEqual(
                            fa["exemplars"],
                            {
                                name: {"direction": "ltr", "unicodeBidi": "isolate"}
                                for name in required_exemplars
                            },
                            page_name,
                        )
                        expected_urls = (
                            [{"direction": "ltr", "unicodeBidi": "isolate", "visible": True}]
                            if page_name == "families-log-density-likelihood.html"
                            else []
                        )
                        self.assertEqual(fa["urlExemplars"], expected_urls, page_name)
                        if page_name in {"api.html", "index.html"}:
                            page.evaluate("window.scrollTo(0, 0)")
                            screenshot = artifact_directory / f"sphinx-{page_name[:-5]}-fa.png"
                            page.screenshot(path=str(screenshot), full_page=True)
                            self.assertGreater(screenshot.stat().st_size, 0)
                        page.goto((outputs["de"] / page_name).as_uri(), wait_until="load")
                        de = page.evaluate("""(requiredExemplars) => ({
                      lang: document.documentElement.lang, dir: document.documentElement.dir,
                      body: getComputedStyle(document.body).direction,
                      exemplars: Object.fromEntries([
                        ['code', 'code.literal'], ['pre', '.highlight pre'],
                        ['table', 'table.docutils'], ['math', '.math'],
                        ['formula', 'p[dir="ltr"][align="center"]'],
                      ].filter(([name]) => requiredExemplars.includes(name))
                        .map(([name, selector]) => {
                        const element = document.querySelector(selector);
                        if (element === null) {
                          throw new Error(`missing required exemplar: ${selector}`);
                        }
                        const style = getComputedStyle(element);
                        return [name, {direction: style.direction, unicodeBidi: style.unicodeBidi}];
                      })),
                        })""", list(required_exemplars))
                        self.assertEqual(de["lang"], "de", page_name)
                        self.assertEqual(de["dir"], "ltr", page_name)
                        self.assertEqual(de["body"], "ltr", page_name)
                        for name in required_exemplars:
                            with self.subTest(page=page_name, exemplar=name):
                                self.assertEqual(de["exemplars"][name]["direction"], "ltr")
                                self.assertIn(
                                    de["exemplars"][name]["unicodeBidi"],
                                    {"normal", "isolate"},
                                )
                finally:
                    browser.close()
