# Veridist: Funktionsleitfaden

[English](capability-guide.md) | [فارسی](capability-guide.fa.md) | [Deutsch](capability-guide.de.md)

Dieser Leitfaden beschreibt die praktische Grenze von **Veridist 1.0.1**: welche Daten das Paket annimmt, welches Ergebnis es liefert und wo die Unterstützung endet. Eine Funktion ist nur unter den genannten Daten- und Ausführungsbedingungen unterstützt.

## Ich möchte die Lebensdauer von Geräten analysieren

Veridist passt eine statistische Verteilung an beobachtete Lebensdauern an. **Exponential-MLE**, **Weibull-Minimum-MLE** und **Lognormal-MLE** sind mit fester Lage `loc=0` für exakte und unabhängige Rechtszensierung verfügbar. Der strikte UTF-8-CSV-Weg ist enger: Er passt nur das ratenbasierte Exponentialmodell an und akzeptiert genau `time,event_observed`.

Weibull-Minimum und Lognormal verwenden typisierte Lebensdauerobjekte. Weibull akzeptiert Häufigkeitsgewichte und optional eine feste Form; Lognormal akzeptiert Häufigkeitsgewichte. Bei Erfolg gibt es eine endliche Schätzung, andernfalls einen typisierten statistischen oder Ausführungsfehler mit Ursache.

## Was ist, wenn einige Geräte noch nicht ausgefallen sind?

Verwenden Sie `event_observed=0`, wenn ein Gerät beim Ende der Beobachtung noch nicht ausgefallen war. Das ist unabhängige Rechtszensierung: Die endgültige Lebensdauer ist unbekannt, aber sie ist länger als die beobachtete Zeit. Links- und Intervallzensierung sowie Trunkierung gehören nicht zum Umfang von 1.0.

## Welches Ergebnis erhalte ich?

Ein Fit enthält geschätzte Parameter, Diagnosen und die Annahmen der Berechnung. Für endliche, positive und unzensierte Exponentialstichproben unterstützt Veridist außerdem **Monte-Carlo-KS/AD/CvM mit erneuter Anpassung**, AIC/BIC, eine Kalibrierungsübersicht und eine adequacy-gesteuerte Auswahl. Gewählt wird der Kandidat mit dem kleinsten AIC, der die definierte Angemessenheitsprüfung besteht; andernfalls lautet das Ergebnis `NONE_ADEQUATE`. Dies ist keine automatische Rangfolge aller Fit-Familien.

## Was kann ich außer dem Fit berechnen?

Normal, Gamma, Weibull-Minimum, Lognormal und Rechts-Gumbel unterstützen skalare Log-Dichte, CDF, Überlebensfunktion, Quantile und vom Aufrufer gesteuerte Zufallsstichproben. Diese Operationen sind skalar: Es gibt keine Array-API, und eine Berechnungsfunktion bedeutet nicht automatisch, dass die Familie auch angepasst werden kann.

## Was geschieht bei großen Daten oder einer Unterbrechung?

Vom Aufrufer bereitgestellte Datenblöcke können nacheinander reduziert werden. Erfolgreiche binary64-Log-Dichtewerte verwenden einen festen O(1)-Reduktionszustand mit einer abschließenden binary64-Rundung. Das ist keine allgemeine Aussage zu RSS oder Durchsatz.

Historische Nachweise decken den strikten exponentiellen CSV-Pfad mit 10 Tausend, 100 Tausend und 1 Million Zeilen sowie mehreren Chunk-Größen ab. Sie beschreiben genau diese Läufe, nicht Geschwindigkeit oder Speicherverbrauch jedes Rechners und Datensatzes.

Für kompatible exponentielle Reduktionen kann ein lokaler SQLite-Checkpoint den Zustand nach einer Unterbrechung bewahren. Vor dem Fortsetzen prüft Veridist Quellrevision, Prüfsumme, Checkpoint-Generation und verarbeitete Bereiche. Dauerhafte Wiederaufnahme ist auf einen Host und sein lokales Dateisystem begrenzt; sie ist keine verteilte Ausführung.

## Was wird noch nicht unterstützt?

Die Version 1.0 unterstützt keine Kovariaten wie Temperatur oder Druck, analytischen Gewichte, freien Lageparameter, allgemeinen Dataframe-/Datenbankadapter, verteilte Checkpoints, Bootstrap-Stabilität der Auswahl oder Inferenz für jede registrierte Familie. Die vollständige Grenze steht unter [bekannte Grenzen](../python/KNOWN_LIMITS.de.md).

## Wie wird die Codequalität geprüft?

Ergebnisse werden mit unabhängigen Referenzen verglichen. Tests decken ungültige Eingaben, Grenzfälle, Unterbrechung und Wiederaufnahme ab und laufen auf Python 3.11 bis 3.14. Das Qualitäts-Gate verlangt mindestens 95 % globale Zeilen- und Zweigabdeckung sowie strengere Schwellen für numerische Module. Kritischer statistischer Code durchläuft zusätzlich ein fail-closed Mutation-Gate.

## Technische Details

Messwerte gelten nur für Adapter, Familie, Arbeitslast, Plattform, Python-Version, Chunk-Grenze und Kandidaten-SHA, die tatsächlich getestet wurden. Nicht unterstützte Kombinationen scheitern ausdrücklich. Das strikte CSV-Beispiel und gerenderte persische RTL-Seiten sind ausführbare CI-Verträge.

[Zurück zum Hauptleitfaden](../README.de.md)
