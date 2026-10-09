# Veridist

**Lebensdauerdaten in nachvollziehbare Aussagen über Zuverlässigkeit verwandeln.**

[![PyPI](https://img.shields.io/pypi/v/veridist.svg)](https://pypi.org/project/veridist/)
[![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%E2%80%933.14-3776AB)](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-guide.de.md)
[![CI](https://img.shields.io/github/actions/workflow/status/alisadeghiaghili/veridist/v1-ci.yml?branch=main&label=CI)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml)
[![Coverage ≥95%](https://img.shields.io/badge/coverage%20requirement-%E2%89%A595%25-blue)](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-guide.de.md)
[![License](https://img.shields.io/badge/license-BUSL--1.1-purple)](https://github.com/alisadeghiaghili/veridist/blob/main/LICENSE)

[English](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.md) | [فارسی](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.fa.md) | [Deutsch](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.de.md)

## Was Veridist ist

Veridist ist eine Python-Bibliothek für **Lebensdauerdaten und Zuverlässigkeitsanalyse**. Sie macht Annahmen, Beobachtungszahlen und Ausführungsinformationen einer Schätzung überprüfbar.
Der Name **Veridist** verbindet *verified* und *distribution*: Verteilungsanpassung mit Überprüfbarkeit. Ein erfolgreich ausgeführter Fit erklärt ein Modell nicht automatisch für angemessen.

Veridist unterstützt heute Exponential-, Weibull-Minimum-, Lognormal-, Gamma-, Normal- und Rechts-Gumbel-Modelle, unabhängig rechtszensierte Beobachtungen, Standardfehler, Konfidenzintervalle und B-Lebensdauern für angepasste Modelle sowie stückweise Likelihood-Reduktion und lokales Fortsetzen kompatibler Exponential-CSV-Läufe.

Der CSV-Einstieg ist bewusst eng: Strenges UTF-8-CSV passt nur ein ratenbasiertes Exponentialmodell an. Jede Familie passt außerdem typisierte Beobachtungsobjekte im Speicher an. Die [Funktionsleitfaden](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-guide.de.md) beschreibt die Grenze der Release-Linie.

## Mit einer Lebensdauerfrage beginnen

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
low, high = fit.uncertainty().confidence_intervals()["rate"]
assert low < fit.rate < high
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

## Warum Wahrscheinlichkeitsverteilungen modellieren?

Daten enthalten Muster, Streuung und seltene Ereignisse. Eine angepasste Wahrscheinlichkeitsverteilung beschreibt dieses Verhalten kompakt, unterstützt Wahrscheinlichkeitsberechnungen und macht Unsicherheit sichtbar.

Wenn Daten für das zuverlässige Trainieren und Bewerten komplexer Modelle wie tiefer neuronaler Netze nicht reichen, können statistische Modelle mit weniger Parametern sinnvoll sein, sofern ihre Annahmen passen. Datenmenge allein entscheidet nicht: Analyseziel, Datenstruktur und Erklärbarkeit zählen ebenso. Verteilungsmodellierung bleibt auch bei großen Daten nützlich.

Eine Referenzverteilung kann **Anomalieerkennung** und **Distribution-Drift-Überwachung** unterstützen. Dafür bleiben Validierung, Entscheidungsschwellen und die Kontrolle falscher Alarme erforderlich. Parameter, Quantile und Überschreitungswahrscheinlichkeiten können später Merkmale für Deep-Learning-Modelle sein, wenn sie ohne Zukunfts- oder Testinformationen geschätzt werden.

## Wo dieselben Ideen helfen können

In der Statistik kann „Lebensdauer“ die Zeit bis zu jedem klar definierten Ereignis bedeuten, nicht nur die Lebensdauer einer Maschine. Vor dem Fit müssen Ereignis, Zeiteinheit, untersuchte Population und Grund für das Beobachtungsende festgelegt werden.

| Bereich | Beispielfrage | Was Veridist heute beitragen kann |
| --- | --- | --- |
| Zuverlässigkeit und Fertigung | Wann fällt eine Pumpe, ein Lager, eine Batterie oder ein Bauteil aus? | Unterstützte Lebensdauermodelle anpassen und Einheiten einbeziehen, die am Beobachtungsende noch funktionierten. |
| Gesundheit und Überlebenszeitanalyse | Wie lange dauert es bis zu Rückfall, Wiederaufnahme oder einem anderen dokumentierten Ereignis? | Exakte und unabhängig rechtszensierte Zeiten mit den unterstützten Lebensdauermodellen analysieren; klinische Interpretation und Kovariatenanpassung liegen außerhalb des aktuellen Pakets. |
| Kredit und Versicherung | Wie lange dauert es bis zu Ausfall, vorzeitiger Rückzahlung oder erstem Schaden? | Kundinnen und Kunden ohne Ereignis bis zum Studienende rechtszensiert abbilden und ein unterstütztes Time-to-Event-Modell anpassen. |
| Betrugserkennung und Cybersicherheit | Ist ein Transaktionsbetrag, Zeitabstand oder eine Anmeldelatenz gegenüber einer vertretbaren Referenzverteilung ungewöhnlich? | Skalare Log-Dichte, Randwahrscheinlichkeit oder ein Quantil als ein Signal in einem separat validierten Erkennungssystem verwenden. Veridist ist kein vollständiger Betrugsklassifikator. |
| Betrieb und Lieferketten | Welche Liefer-, Reparatur-, Warte- oder Servicezeit ist zu erwarten? | Positive Dauern mit einem unterstützten Modell beschreiben und Wahrscheinlichkeiten oder Quantile berechnen, wenn Familie und Parameter verfügbar sind. |
| Digitale Produkte und Kundenanalyse | Wie lange dauert es bis zu Abwanderung, Konversion oder einem anderen Produktereignis? | Nutzerinnen und Nutzer, die am Beobachtungsstichtag aktiv bleiben, rechtszensiert behandeln, sofern die Zensierungsannahme vertretbar ist. |

Diese Beispiele teilen eine statistische Struktur, nicht dieselbe fachliche Bedeutung. Fachliche Validierung, Stichprobendesign, Kosten, Entscheidungsschwellen sowie rechtliche oder sicherheitsbezogene Anforderungen bleiben Teil der Anwendung.

## Wenn die Verteilung unbekannt ist

Distribution Fitting kann Kandidaten anpassen, ihre Parameter schätzen und ihre Eignung vergleichen. Der beste Rang ist nicht automatisch die wahre datenerzeugende Verteilung; möglicherweise ist kein Kandidat ausreichend.

Automatische Rangfolgen zwischen den aktuellen Fit-Familien sind ein Zukunftsziel. Die vorhandene Inferenz und angemessenheitsgesteuerte Auswahl gelten enger: für endliche, positive, unzensierte Exponentialstichproben. Die [Funktionsleitfaden](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-guide.de.md) dokumentiert den genauen Vertrag.

## Mit eigenen Daten arbeiten

Übergeben Sie der Fit-Funktion Ihren CSV-Pfad. Der öffentliche CSV-Einstieg ist
nur für Exponential und akzeptiert striktes UTF-8-Lebensdauer-CSV.

| Einstellung | Zweck |
| --- | --- |
| `CsvLifetimeSchema` | Benennt Zeit- und Ereignisindikatorspalten. |
| `PublicSourceId` | Liefert eine nicht geheime Quellkennung in der Ausführungsprovenienz. |
| `CsvLifetimeLimits` | Deklariert Eingabe-Bytebudgets. |

Die [API-Referenz](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/source/api.de.md) erklärt zulässige Eingaben, Ergebnistypen und
typisierte Fehler.

## Modelle und Werkzeuge heute

### Fitting

| Modell | Beschreibbares Muster |
| --- | --- |
| Exponential | Konstante Ausfallrate. |
| Weibull-Minimum | Fallende, konstante oder steigende Ausfallrate, abhängig von der Form. |
| Lognormal | Positive Lebensdauern, deren Logarithmen normal modelliert werden. |
| Gamma | Positive Lebensdauern mit flexibler, rechtsschiefer Form. |
| Normal | Reelle Messwerte um einen Mittelwert. |
| Rechts-Gumbel | Reelle Maxima und andere Extremwerte. |

Die Lebensdauerfamilien (Exponential, Weibull-Minimum, Lognormal und Gamma) verwenden festen Ort null und nehmen exakte sowie unabhängig rechtszensierte Lebensdauern an; Normal und Rechts-Gumbel nehmen exakte und rechtszensierte reelle Werte an. `fit(family, observations)` verteilt auf eine Familie, und jede Familie hat außerdem eine eigene Funktion wie `fit_weibull`. Das CSV-Beispiel oben passt nur Exponential an. Jede erfolgreiche Anpassung hat eine Methode `uncertainty()`, die Standardfehler, Konfidenzintervalle und abgeleitete Größen wie Mittelwert, B-Lebensdauer oder Überlebenswahrscheinlichkeit liefert.

### Wahrscheinlichkeitsberechnungen

`logpdf`, `cdf`, `sf`, `ppf` und `sample` nehmen die Familie zuerst und die Parameter als Schlüsselwörter. Sie decken alle sechs Familien ab und akzeptieren Skalare und numpy-Arrays; `lifetimes_from_arrays` und `values_from_arrays` erzeugen Beobachtungen aus Spalten. Siehe den [Familien- und Likelihood-Leitfaden](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/source/families-log-density-likelihood.md); beim Umstieg von Version 1.0 lesen Sie die [Migrationsanleitung](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/migration-2.0.md).

### Modellbewertung

Endliche positive unzensierte Exponentialstichproben unterstützen Refit-Monte-
Carlo-KS/AD/CvM, AIC/BIC und adequacy-gesteuerte Auswahl mit einem
aufrufereigenen Generator. Inferenz ist enger als Fitting; die
[Capability Guide](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-guide.de.md)
dokumentiert den genauen Umfang.

Das frühere Projekt enthielt 25 Verteilungen: 20 stetige und fünf diskrete.
Ihr Migrationsstatus ist nicht gleich dem veröffentlichten Umfang; maßgeblich ist
der Capability Guide.

## Wenn Daten wachsen

Likelihood-Werkzeuge reduzieren vom Aufrufer gelieferte Chunks, mit oder ohne Rechtszensierung (`reduce_lifetime_log_likelihood_chunks` und `reduce_value_log_likelihood_chunks`); Ihre Anwendung
entscheidet, wie Daten geteilt und geliefert werden. `SQLiteCheckpointStore`
speichert lokalen Neustartzustand für kompatible Exponential-Reduktionen,
einschließlich des unterstützten CSV-Pfads. Halten Sie die Quellrevision stabil
und folgen Sie dem [Checkpoint- und Fortsetzungsrezept](https://github.com/alisadeghiaghili/veridist/blob/main/python/examples/checkpoint_resume.py).

Erhaltene Skalierungsnachweise für den strikten CSV-/Exponentialpfad decken
10k, 100k und 1m Zeilen bei Chunk-Grenzen von 32, 64 und 128 KiB ab. Sie
stammen aus je einem Lauf auf einem Linux- und einem Windows-Runner
(Kandidaten-Commit `19ecf10`, 2026-10-09, CPython 3.11): 1m Zeilen benötigten
14 bis 20 Sekunden bei einem Spitzenwert des Prozessspeichers von etwa 40 bis
45 MiB. Diese Werte beschreiben nur diese Läufe; sie sind keine Geschwindigkeits-
oder Speichergarantie für andere Maschinen oder Daten. Dauerhafte
Wiederaufnahme läuft auf einer Maschine mit lokalem Dateisystem.

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
`distfit_pro`-Quellcode ist nicht Teil von Veridist und keine
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
- **Breitere statistische Bewertung:** Gütewerkzeuge auf
  weitere Familien und Beobachtungsbedingungen ausdehnen.
- **Effizientere Verarbeitung großer Daten:** Laufzeit und Speicher mit
  reproduzierbaren Experimenten messen und verbessern.
- **Praktische Vignetten:** Von einer realen Frage über Daten zur Interpretation
  führen und Verteilungsmerkmale für Anomalien, Drift und Machine Learning
  untersuchen.

Dies sind Entwicklungsrichtungen, keine aktuell unterstützten Funktionen oder
zugesagten Veröffentlichungstermine. Veröffentlichte Fähigkeiten stehen in der
[Capability Guide](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-guide.de.md),
gelieferte Änderungen im [Changelog](https://github.com/alisadeghiaghili/veridist/blob/main/python/CHANGELOG.md).

## Leitfäden, Hilfe und Beiträge

| Ihr Ziel | Wohin |
| --- | --- |
| Eigenständigen Paketleitfaden lesen | [Package README](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.de.md) |
| Zensierungsbeispiel lernen | [Exponential-Leitfaden](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/source/exponential-right-censoring.md) |
| Eingaben, Ergebnisse und Fehler prüfen | [API-Referenz](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/source/api.de.md) |
| Von Version 1.0 umsteigen | [Migrationsanleitung](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/migration-2.0.md) |
| Reproduzierbaren Defekt melden | [GitHub Issues](https://github.com/alisadeghiaghili/veridist/issues) |
| Beitragen | [Beitragsleitfaden](https://github.com/alisadeghiaghili/veridist/blob/main/CONTRIBUTING.md) |
| Sicherheitsproblem melden | [Sicherheitsrichtlinie](https://github.com/alisadeghiaghili/veridist/blob/main/SECURITY.md) |
| Releases verfolgen | [Changelog](https://github.com/alisadeghiaghili/veridist/blob/main/python/CHANGELOG.md) |

## Veridist zitieren

Zitieren Sie die Release, die Ihr Ergebnis erzeugt hat. Der
[Zitierleitfaden](https://github.com/alisadeghiaghili/veridist/blob/main/docs/citing-veridist.md)
bietet IEEE, APA 7, Chicago, MLA 9, Harvard, Vancouver, BibTeX, RIS, EndNote
XML und CSL-JSON. [CITATION.cff](https://github.com/alisadeghiaghili/veridist/blob/main/CITATION.cff)
ist der kanonische maschinenlesbare Eintrag.

## Autor- und Forschungsprofile

Maintainer ist [Seyed Ali Sadeghi Aghili](https://linktr.ee/aliaghili).

[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=BDD_JUIAAAAJ&hl=en&authuser=1)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-00CCBB?logo=researchgate&logoColor=white)](https://www.researchgate.net/profile/Seyed-Ali-Sadeghi-Aghili)
[![PeerJ](https://img.shields.io/badge/PeerJ-00A4A6?logo=peerj&logoColor=white)](https://peerj.com/AliSadeghiAghili/)
[![ORCID](https://img.shields.io/badge/ORCID-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0002-5938-3291)

## Fachbegriffe

Wichtige Begriffe werden auch auf Englisch genannt: Distribution Fitting,
Likelihood, Right Censoring, Anomaly Detection und Distribution Drift.

## Lizenz

Veridist wird unter **Business Source License 1.1 (BUSL-1.1)** vertrieben.
Der Quelltext ist einsehbar, die Software ist jedoch nicht Open Source.
[LICENSE](https://github.com/alisadeghiaghili/veridist/blob/main/LICENSE) erlaubt
die produktive Nutzung nur für nichtkommerzielle Zwecke: private Nutzung,
akademische Forschung und Lehre sowie Nutzung durch gemeinnützige
Organisationen für ihre nichtkommerziellen Aktivitäten. Jede andere produktive
Nutzung, auch durch oder im Auftrag eines gewinnorientierten Unternehmens und
die interne betriebliche Nutzung, erfordert eine kommerzielle Lizenz des
Lizenzgebers (alisadeghiaghili@gmail.com). Zum Umstellungsdatum 2030-09-05,
oder zum vierten Jahrestag der ersten öffentlichen Verbreitung einer Version,
falls dieser früher liegt, wechselt die Lizenz zu Apache License, Version 2.0.
Der BUSL-1.1-Badge bedeutet nicht, dass die aktuelle Release heute unter
Apache-2.0 lizenziert ist.
