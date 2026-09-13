# Veridist: Funktionsübersicht

[English](capability-matrix.md) | [فارسی](capability-matrix.fa.md) | [Deutsch](capability-matrix.de.md)

Diese Übersicht beschreibt die aufrufbaren Funktionen von **Veridist 1.0.1**. Jede Funktion wird nur innerhalb der genannten Daten- und Ausführungsbedingungen unterstützt. Historische Nachweise liefern Kontext; Aussagen zu einer Veröffentlichung benötigen Prüfungen der konkreten Revision.

## Modellanpassung und Statistik

| Funktion | Unterstützter Umfang | Ergebnis oder Fehler | Nachweise |
| --- | --- | --- | --- |
| Exponential-MLE | Feste Lage null, nur Rate; exakte Zeiten und unabhängige Rechtszensierung; typisierte Lebensdauerobjekte oder striktes UTF-8-CSV | Endliche Punktschätzung oder typisierter statistischer beziehungsweise Ausführungsfehler | Referenz-, Vertrags-, CSV-, Checkpoint-, Abdeckungs- und Mutationstests |
| Weibull-Minimum-MLE | Feste Lage null; exakte Zeiten und unabhängige Rechtszensierung; optionale Häufigkeitsgewichte und optional feste Form | Endliche Form-/Skalenschätzung oder typisierter Fehler mit Konvergenzinformationen | Unabhängige Referenzen und Verträge für Zensierung und Gewichte |
| Lognormal-MLE | Feste Lage null; exakte Zeiten und unabhängige Rechtszensierung; optionale Häufigkeitsgewichte | Endliche logarithmische Lage-/Skalenschätzung oder typisierter Fehler mit Konvergenzinformationen | Unabhängige Referenzen und Verträge für Zensierung und Gewichte |
| Skalare Verteilungsoperationen | Normal, Gamma, Weibull-Minimum, Lognormal und Rechts-Gumbel | Log-Dichte, CDF, Überlebensfunktion, Quantil und Stichproben mit einem vom Aufrufer verwalteten Zufallsgenerator | Konformitäts-, Identitäts-, Randfall- und Abdeckungstests; nur skalar |
| Inferenz und Auswahl | Endliche, positive, unzensierte Exponentialstichproben | Monte-Carlo-KS/AD/CvM mit erneuter Anpassung, AIC/BIC, Kalibrierungsübersicht; kleinstes AIC unter geeigneten Kandidaten oder `NONE_ADEQUATE` | NumPy-Generator des Aufrufers; angeforderte, erfolgreiche und fehlgeschlagene Neuanpassungen sowie Monte-Carlo-Unsicherheit werden berichtet |

## Schrittweise Verarbeitung und Wiederaufnahme

| Funktion | Unterstützter Umfang | Ergebnis oder Fehler | Nachweise |
| --- | --- | --- | --- |
| Streaming-Likelihood | Exakte Akkumulation erfolgreicher skalarer binary64-Log-Dichtewerte | Einmalige abschließende Rundung auf binary64; explizite vorzeichenlose 64-Bit-Zählgrenze | Vertrags- und generierte Stream-Nachweise; keine allgemeine RSS- oder Durchsatzzusage |
| Dauerhafte Wiederaufnahme | Sequenzielle Exponentialreduktion kanonischer Chunks oder strikter Lebensdauer-CSV; ein Host und lokales Dateisystem | SQLite-Generationsvergleich mit atomarem Austausch, Prüfsummen-, Quellrevisions- und Bereichsprüfungen | Ende-zu-Ende-Tests für Unterbrechung, Wiederholung, Beschädigung, konkurrierende Zugriffe und Abbruch |

Der konstante Reduktionszustand mit `O(1)` und das logische Lieferbudget sind algorithmische Verträge. Sie belegen weder eine portable RSS-Obergrenze noch universellen Durchsatz, verteilte Ausführung oder allgemeine Verarbeitung von Daten, die größer als der Arbeitsspeicher sind.

## Versionsgrenzen

Nicht unterstützte Kombinationen scheitern ausdrücklich. Die 1.0-Linie bietet keine Links- oder Intervallzensierung, Trunkierung, Kovariaten, analytischen Gewichte, freien Lageparameter, Array-API, verteilte Checkpoint-Ablage, allgemeinen Dataframe-/Datenbankadapter, Bootstrap-Auswahlstabilität oder Inferenz für jede registrierte Familie.

Die englischen, persischen und deutschen READMEs beschreiben dieselbe Veröffentlichungsgrenze. Das strikte CSV-Beispiel und die gerenderten persischen RTL-Berichte werden in CI geprüft. Einzelheiten stehen unter [bekannte Grenzen](../python/KNOWN_LIMITS.de.md).

## Qualität und Nachweise

Die Veröffentlichungslinie wird auf Python 3.11 bis 3.14 getestet. Das Abdeckungsmanifest verlangt mindestens 95 % globale Zeilen- und Zweigabdeckung sowie strengere Schwellen für numerische Module. Der kritische statistische Kern muss außerdem ein Mutations-Gate bestehen; bei dessen Fehlschlag wird die Prüfung nicht akzeptiert. Kandidatenspezifische Skalierungs- und Veröffentlichungsprüfungen binden gespeicherte Nachweise an einen vollständigen Commit-SHA.

[Zurück zum Hauptleitfaden](../README.de.md)
