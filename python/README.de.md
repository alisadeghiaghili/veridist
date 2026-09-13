# Veridist

**Lebensdauerdaten in nachvollziehbare Aussagen über Zuverlässigkeit verwandeln.**

[![PyPI](https://img.shields.io/pypi/v/veridist.svg)](https://pypi.org/project/veridist/)
[![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%E2%80%933.14-3776AB)](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md)
[![CI](https://img.shields.io/github/actions/workflow/status/alisadeghiaghili/veridist/v1-ci.yml?branch=main&label=CI)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml)
[![Coverage ≥95%](https://img.shields.io/badge/coverage%20requirement-%E2%89%A595%25-blue)](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md)
[![License](https://img.shields.io/badge/license-BUSL--1.1-purple)](https://github.com/alisadeghiaghili/veridist/blob/main/LICENSE)

[English](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.md) | [فارسی](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.fa.md) | [Deutsch](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.de.md)

## Warum Wahrscheinlichkeitsverteilungen modellieren?

Daten enthalten Muster, typische Ergebnisse, Streuung und seltene Ereignisse.
Das Anpassen einer Wahrscheinlichkeitsverteilung hilft, diese Muster zu
beschreiben, Ereigniswahrscheinlichkeiten zu schätzen und Unsicherheit zu
berücksichtigen.

Wenn es zu wenig Daten gibt, um komplexe Modelle wie tiefe neuronale Netze
zuverlässig zu trainieren und zu bewerten, können statistische Modelle mit
weniger Parametern sinnvoll sein — sofern ihre Annahmen zum Problem passen. Sie
ermöglichen eine erklärbare Darstellung statistischen Verhaltens mit begrenzten
Beobachtungen. Datenmenge allein entscheidet jedoch nicht über die Methode:
Analyseziel, Datenstruktur und Erklärbarkeit zählen ebenfalls. Auch bei großen
Daten bleibt Verteilungsmodellierung nützlich.

Eine Referenzverteilung kann ungewöhnliche Beobachtungen oder Änderungen im
Muster neuer Daten sichtbar machen. Das sind Grundlagen für
**Anomalieerkennung** und **Distribution-Drift-Überwachung**. Verlässliche
Anwendungen brauchen außerdem Modellvalidierung, Entscheidungsschwellen und die
Kontrolle falscher Alarme.

Verteilungsparameter, Quantile und Überschreitungswahrscheinlichkeiten können
später auch Merkmale für Deep-Learning-Modelle sein. Ihr Nutzen muss auf
separaten Auswertungsdaten geprüft werden; die Merkmale dürfen weder
Zukunfts- noch Testinformationen verwenden, damit kein Datenleck entsteht.

## Wenn die Verteilung unbekannt ist

Distribution Fitting kann mehrere Kandidaten an Daten anpassen, ihre Parameter
schätzen und vergleichen, wie gut sie Beobachtungen beschreiben. Ein
Mehrmodell-Workflow kann Rangfolgen mit Parametern und Bewertungsmaßen liefern.

Der beste Rang ist nicht automatisch die wahre datenerzeugende Verteilung.
Kandidatenmenge, Vergleichskriterium und Annahmen bestimmen die Aussagekraft;
möglicherweise ist kein Kandidat ausreichend. Veridist bietet derzeit getrennte
APIs für Exponential-, Weibull-Minimum- und Lognormal-Fits. Automatische
Rangfolgen zwischen Familien sind ein Zukunftsziel, keine aktuelle Funktion.

## Was Veridist leistet

Veridist ist eine Python-Bibliothek für **Lebensdauerdaten und
Zuverlässigkeitsanalyse**. Sie hilft Ingenieurinnen, Ingenieuren und Forschenden,
Beobachtungen in Schätzungen zu überführen, deren Annahmen und
Ausführungsinformationen überprüfbar sind.

Der Name **Veridist** verbindet *verified* und *distribution*: Verteilungsanpassung, die auf Überprüfbarkeit ausgelegt ist.

- Exponential-, Weibull-Minimum- und Lognormal-Lebensdauermodelle anpassen.
- Beobachtungen einbeziehen, die vor dem Ereignis endeten.
- Schätzungen, Beobachtungszahlen und Berechnungsannahmen prüfen.
- Skalare Verteilungsoperationen und stückweise Likelihood-Reduktion nutzen.
- Kompatible unterbrochene Berechnungen auf unterstützten lokalen Pfaden fortsetzen.

Anomalieerkennung, Drift-Überwachung und Deep Learning sind weitergehende
Anwendungen der Verteilungsmodellierung. Veridist liefert dafür derzeit keine
fertigen Systeme.

## Von der Beobachtung zum Modell

Stellen Sie sich eine Flotte von Pumpen vor. Für jede Pumpe kennen Sie die
Beobachtungszeit und wissen, ob sie ausgefallen ist.

| Pumpenstatus | Aussage der Beobachtung |
| --- | --- |
| Während der Studie ausgefallen | Die Ausfallzeit ist bekannt. |
| Bei Beobachtungsende noch in Betrieb | Die Lebensdauer übersteigt die Beobachtungszeit. |

Der zweite Fall heißt **Rechtszensierung**: Die Beobachtung endete, bevor der
Ausfall zu sehen war. Er liefert trotzdem Information. Ein statistisches Modell
ist eine vereinfachte Beschreibung dieser Zeiten; ob seine Annahmen die Anlage
beschreiben, bleibt Teil Ihrer Analyse.

## Ihre erste Analyse

Wir beginnen mit einem Exponentialmodell. Es nimmt eine konstante Ausfallrate
an: ein einfaches Lernbeispiel, aber nicht zwingend ein passendes Modell für
alternde Anlagen.

### Installieren

Python 3.11 bis 3.14 werden unterstützt.

```console
python -m pip install veridist
```

### Daten verstehen

| time | event_observed | Bedeutung |
| --- | --- | --- |
| 1 | 1 | Ein Ausfall wurde zur Zeit 1 beobachtet. |
| 1 | 0 | Die Pumpe lief zur Zeit 1 noch. |

Wählen Sie eine Zeiteinheit, etwa Stunden oder Monate, und verwenden Sie sie
durchgängig. Die zwei Zeilen erklären die Rechnung, reichen aber nicht für eine
reale Zuverlässigkeitsentscheidung.

### Modell anpassen

Das Beispiel erzeugt seine CSV selbst und läuft nach der Installation.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from veridist import (
    CsvLifetimeLimits,
    CsvLifetimeSchema,
    PublicSourceId,
    fit_exponential_csv,
)
from veridist.families import ExponentialFitSuccess

with TemporaryDirectory() as directory:
    path = Path(directory) / "lifetimes.csv"
    path.write_text("time,event_observed\n1,1\n1,0\n", encoding="utf-8")
    fit = fit_exponential_csv(
        path,
        schema=CsvLifetimeSchema("time", "event_observed"),
        source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
        limits=CsvLifetimeLimits(32768, 32768),
    ).fit
assert isinstance(fit, ExponentialFitSuccess)
assert fit.rate == 0.5
assert fit.inference == "not_provided"
assert fit.censoring_assumption == "independent_right_censoring"
print(f"rate={fit.rate}; events={fit.event_count}; censored={fit.censored_count}")
```

```text
rate=0.5; events=1; censored=1
```

### Ergebnis interpretieren

Ein Ausfall über zwei beobachtete Zeiteinheiten ergibt die geschätzte Rate
`1 / 2 = 0.5`. Bei Monaten ist das 0.5 Ausfälle pro Pumpenmonat, nicht eine
Ausfallwahrscheinlichkeit von 50 % in einem Monat.

Die noch arbeitende Pumpe liefert eine beobachtete Zeiteinheit ohne Ausfall.
Das Modell nimmt an, dass Zensierung von der unbeobachteten Ausfallzeit
unabhängig ist. Ein erfolgreicher Fit bestätigt die Berechnung, nicht die
Angemessenheit des Modells. Folgen Sie dem
[Leitfaden zur Rechtszensierung](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/source/exponential-right-censoring.md).

## Mit eigenen Daten arbeiten

Übergeben Sie der Fit-Funktion Ihren CSV-Pfad. Der öffentliche CSV-Einstieg ist
nur für Exponential und akzeptiert striktes UTF-8-Lebensdauer-CSV.

| Einstellung | Zweck |
| --- | --- |
| `CsvLifetimeSchema` | Benennt Zeit- und Ereignisindikatorspalten. |
| `PublicSourceId` | Liefert eine nicht geheime Quellkennung in der Ausführungsprovenienz. |
| `CsvLifetimeLimits` | Deklariert Eingabe-Bytebudgets. |

Die [API-Referenz](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/source/api.md) erklärt zulässige Eingaben, Ergebnistypen und
typisierte Fehler.

## Modelle und Werkzeuge heute

### Lebensdauer-Fitting

| Modell | Beschreibbares Muster |
| --- | --- |
| Exponential | Konstante Ausfallrate. |
| Weibull-Minimum | Fallende, konstante oder steigende Ausfallrate, abhängig von der Form. |
| Lognormal | Positive Lebensdauern, deren Logarithmen normal modelliert werden. |

Diese Fits verwenden festen Ort null und unterstützen exakte sowie unabhängig
rechtszensierte Lebensdauern. Weibull-Minimum und Lognormal verwenden ihre
Modell-APIs; das CSV-Beispiel passt nur Exponential an.

### Wahrscheinlichkeitsberechnungen

Skalare Operationen für Normal-, Gamma-, Weibull-Minimum-, Lognormal- und
Rechts-Gumbel-Familien umfassen Log-Dichte, CDF, Survival, Quantile und
Sampling mit aufrufereigenem RNG. Eine verfügbare Verteilungsoperation bedeutet
nicht, dass eine Fit-API verfügbar ist. Siehe
[Familien- und Likelihood-Leitfaden](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/source/families-log-density-likelihood.md).

### Modellbewertung

Endliche positive unzensierte Exponentialstichproben unterstützen Refit-Monte-
Carlo-KS/AD/CvM, AIC/BIC und adequacy-gesteuerte Auswahl mit einem
aufrufereigenen Generator. Inferenz ist enger als Fitting; die
[Capability Matrix](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md)
dokumentiert den genauen Umfang.

Das frühere Projekt enthielt 25 Verteilungen: 20 stetige und fünf diskrete.
Ihr Migrationsstatus ist nicht gleich dem veröffentlichten Umfang; siehe
[Migrationsübersicht](https://github.com/alisadeghiaghili/veridist/blob/main/docs/migration/README.md).

## Wenn Daten wachsen

Likelihood-Werkzeuge reduzieren vom Aufrufer gelieferte Chunks; Ihre Anwendung
entscheidet, wie Daten geteilt und geliefert werden. `SQLiteCheckpointStore`
speichert lokalen Neustartzustand für kompatible Exponential-Reduktionen,
einschließlich des unterstützten CSV-Pfads. Halten Sie die Quellrevision stabil
und folgen Sie dem [Checkpoint- und Fortsetzungsrezept](https://github.com/alisadeghiaghili/veridist/blob/main/python/examples/checkpoint_resume.py).

Release-Evidenz deckt festgelegte CSV-/Exponentialpfade bei 10k, 100k und 1m
Zeilen unter dokumentierten Bedingungen ab. Sie belegt keinen universellen
Durchsatz oder eine portable Prozessspeichergrenze. Dauerhafte Wiederaufnahme
läuft auf einer Maschine mit lokalem Dateisystem.

## Wie Qualität geprüft wird

| Prüfung | Was sie belegt |
| --- | --- |
| Statistische Referenz- und API-Vertragstests | Numerische Ergebnisse, Grenzen und explizites Fehlerverhalten. |
| Coverage ≥95 % | Erforderliche globale Zeilen- und Branch-Coverage; Kernmodule haben strengere Schwellen. |
| Mutationstests des statistischen Kerns | Ob Tests absichtlich fehlerhafte Änderungen erkennen. |
| Paketbau und Installationsprüfungen | Ob Release-Artefakte gebaut und installiert werden können. |
| Ausführbare Beispiele und mehrsprachige Dokumentation | Ob Beispiele laufen und Dokumentation gebaut wird. |

Der Coverage-Badge nennt die geforderte Schwelle, keinen gemessenen aktuellen
Prozentsatz. Der CI-Badge meldet den Status des Hauptworkflows.

## Vor dem Einsatz von Veridist

Prüfen Sie, ob statistische Annahmen zur Datenerhebung passen, ob der benötigte
Fit- und Inferenzpfad verfügbar ist und ob lokale Verarbeitung und
Wiederaufnahme zu Ihren Arbeitslasten passen.

Die [bekannten Grenzen](https://github.com/alisadeghiaghili/veridist/blob/main/python/KNOWN_LIMITS.de.md) nennen Ausschlüsse wie Links- und
Intervallzensierung, Kovariaten und verteilte Ausführung. Der historische
`distfit_pro`-Quellcode wird in der Migrationsübersicht geführt, ist aber keine
Laufzeitkompatibilitätszusage.

Die unteren API-Bausteine \`FAMILY_REGISTRY\`, \`evaluate_log_density\` und
\`reduce_log_likelihood_chunks\` dienen der Familienauswahl, der skalaren
Log-Dichte und der stückweisen Likelihood-Reduktion.

## Zukunftspläne

Die Entwicklung richtet sich auf breitere Modellabdeckung, einfachere Analyse
und statistische Korrektheit:

- **Die 25 historischen Verteilungen prüfen und migrieren:** 20 stetige und
  fünf diskrete Verteilungen mit numerischen Tests, Dokumentation und klaren
  Unterstützungsgrenzen.
- **Mehrmodell-Fitting und -Vergleich:** Rangfolgen mit Parametern,
  Vergleichsmaßen, Angemessenheitsinformation und einem expliziten Ergebnis,
  wenn kein Modell passt.
- **Breitere statistische Bewertung:** Güte- und Unsicherheitswerkzeuge auf
  weitere Familien und Beobachtungsbedingungen ausdehnen.
- **Effizientere Verarbeitung großer Daten:** Laufzeit und Speicher mit
  reproduzierbaren Experimenten messen und verbessern.
- **Praktische Vignetten:** Von einer realen Frage über Daten zur Interpretation
  führen und Verteilungsmerkmale für Anomalien, Drift und Machine Learning
  untersuchen.

Dies sind Entwicklungsrichtungen, keine aktuell unterstützten Funktionen oder
zugesagten Veröffentlichungstermine. Veröffentlichte Fähigkeiten stehen in der
[Capability Matrix](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md),
gelieferte Änderungen im [Changelog](https://github.com/alisadeghiaghili/veridist/blob/main/python/CHANGELOG.md).

## Leitfäden, Hilfe und Beiträge

| Ihr Ziel | Wohin |
| --- | --- |
| Eigenständigen Paketleitfaden lesen | [Package README](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.de.md) |
| Zensierungsbeispiel lernen | [Exponential-Leitfaden](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/source/exponential-right-censoring.md) |
| Eingaben, Ergebnisse und Fehler prüfen | [API-Referenz](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/source/api.md) |
| Reproduzierbaren Defekt melden | [GitHub Issues](https://github.com/alisadeghiaghili/veridist/issues) |
| Beitragen | [Beitragsleitfaden](https://github.com/alisadeghiaghili/veridist/blob/main/CONTRIBUTING.md) und [Engineering-Konventionen](https://github.com/alisadeghiaghili/veridist/blob/main/docs/conventions.md) |
| Sicherheitsproblem melden | [Sicherheitsrichtlinie](https://github.com/alisadeghiaghili/veridist/blob/main/SECURITY.md) |
| Releases verfolgen | [Changelog](https://github.com/alisadeghiaghili/veridist/blob/main/python/CHANGELOG.md) |

## Veridist zitieren

Zitieren Sie die Release, die Ihr Ergebnis erzeugt hat. Der
[Zitierleitfaden](https://github.com/alisadeghiaghili/veridist/blob/main/docs/citing-veridist.md)
bietet IEEE, APA 7, Chicago, MLA 9, Harvard, Vancouver, BibTeX, RIS, EndNote
XML und CSL-JSON. [CITATION.cff](https://github.com/alisadeghiaghili/veridist/blob/main/CITATION.cff)
ist der kanonische maschinenlesbare Eintrag.

## Autor- und Forschungsprofile

Maintainer ist [Seyed Ali Sadeghi Aghili](https://zil.ink/thedatascientist).
[Google Scholar](https://scholar.google.com/citations?user=BDD_JUIAAAAJ&hl=en&authuser=1) ·
[ResearchGate](https://www.researchgate.net/profile/Seyed-Ali-Sadeghi-Aghili) ·
[PeerJ](https://peerj.com/AliSadeghiAghili/)
- [ORCID](https://orcid.org/0000-0002-5938-3291)

## Fachbegriffe

Wichtige Begriffe werden auch auf Englisch genannt: Distribution Fitting,
Likelihood, Right Censoring, Anomaly Detection und Distribution Drift.

## Lizenz

Veridist wird unter **Business Source License 1.1 (BUSL-1.1)** vertrieben.
[LICENSE](https://github.com/alisadeghiaghili/veridist/blob/main/LICENSE) legt
die bedingte zusätzliche Apache-2.0-Nutzungserlaubnis und das Umstellungsdatum
fest. Der BUSL-1.1-Badge bedeutet nicht, dass die aktuelle Release
uneingeschränkt unter Apache-2.0 lizenziert ist.
