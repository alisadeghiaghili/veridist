<a id="veridist-api-de"></a>
# Veridist-API-Leitfaden

<a href="api.md">Englisch</a> | <a href="api.fa.md">Persisch</a> | <a href="api.de.md">Deutsch</a>

Diese Anleitung erklärt die aktuelle öffentliche API von Veridist 2.0. Um eine Verteilung<sup id="fnref-fitting"><a href="#fn-fitting">1</a></sup> an Lebensdauer- oder Messdaten anzupassen, übergeben Sie in der Regel Ihre Beobachtungen oder eine CSV-Datei an eine Anpassungsfunktion und lesen das zurückgegebene Ergebnis. Sechs Verteilungsfamilien haben eine Anpassung, jede Verteilungsoperation akzeptiert auch numpy-Arrays, und jede erfolgreiche Anpassung kann die Unsicherheit ihrer Schätzung angeben<sup id="fnref-uncertainty"><a href="#fn-uncertainty">22</a></sup>. Wenn eine Berechnung lange dauert und unterbrochen werden kann, können Sie den Fortschritt speichern und später fortsetzen. Skalare Werkzeuge<sup id="fnref-scalar"><a href="#fn-scalar">2</a></sup> und vom Aufrufer verwaltete Datenströme<sup id="fnref-caller"><a href="#fn-caller">3</a></sup> stehen außerdem für technischere Anwendungen bereit. Wenn Sie von Version 1.0 umsteigen, lesen Sie die <a href="../migration-2.0.md">Migrationsanleitung</a>.

## Denselben Zeit-Ereignis-Vertrag in verschiedenen Bereichen verwenden

Die beiden CSV-Spalten beschreiben einen allgemeinen Time-to-Event-Datensatz. `time` gibt an, wie lange das Element beobachtet wurde. `event_observed` gibt an, ob das gewählte Ereignis in dieser Zeit eingetreten ist. Der Code entscheidet nicht, was das Ereignis bedeutet; Ihre Analyse muss es einheitlich definieren.

| Anwendungsbereich | Mögliche Bedeutung von `time` | Mögliche Bedeutung von Ereignis `1` | Mögliche Bedeutung von Ereignis `0` |
| --- | --- | --- | --- |
| Zuverlässigkeitsdatensatz | Beobachtungsstunden eines Bauteils | Das Bauteil fiel aus | Es funktionierte am Beobachtungsende noch |
| Datensatz der Gesundheitsforschung | Tage der Nachbeobachtung einer Person | Das definierte Gesundheitsereignis trat ein | Bis zur letzten Nachbeobachtung trat es nicht ein |
| Kredit- oder Versicherungsdatensatz | Monate der Beobachtung eines Kontos oder einer Police | Ausfall oder erster Schaden trat ein | Bis zum Studienende trat kein solches Ereignis ein |
| Datensatz eines digitalen Produkts | Tage seit Registrierung oder Kampagneneintritt | Abwanderung oder Konversion trat ein | Der Nutzer blieb bis zum Stichtag ohne Ereignis aktiv |
| Betriebsdatensatz | Stunden seit Eröffnung einer Reparatur, Bestellung oder eines Falls | Der Vorgang wurde abgeschlossen | Der Vorgang war am Ende der Datenerhebung noch offen |

Für Betrugserkennung oder Cybersicherheit kann die Time-to-Event-CSV zu Fragen wie der Zeit bis zu einem Alarm passen, ist aber kein allgemeines Betrugsdatenformat. Die grundlegenden Verteilungswerkzeuge können außerdem einen Transaktionsbetrag, Zeitabstand oder eine Latenz mit einer bereits festgelegten Referenzverteilung vergleichen. Ein solcher Wert ist ein Eingangssignal für ein separat getestetes Erkennungssystem, keine Betrugsentscheidung für sich allein.

## Ausführungspfad wählen

<table width="100%">
  <thead>
    <tr>
      <th width="36%" align="center"><p align="center">Was Sie tun möchten</p></th>
      <th width="32%" align="center"><p align="center">Funktion</p></th>
      <th width="32%" align="center"><p align="center">Grenze dieses Pfads</p></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Eine CSV-Datei lesen und ein Exponentialmodell in einem Lauf anpassen</td>
      <td><code class="literal">fit_exponential_csv</code></td>
      <td>Bietet kein Fortsetzen, keinen Abbruch und keine automatische Modellauswahl.</td>
    </tr>
    <tr>
      <td>Eine Berechnung mit segmentierten JSON-Daten fortsetzen</td>
      <td><code class="literal">fit_exponential_checkpointed_chunks</code></td>
      <td>Ihr Programm muss die Datenabschnitte vorbereiten und lesen.</td>
    </tr>
    <tr>
      <td>Eine CSV-Datei mit Abbruchmöglichkeit verarbeiten und ab der letzten gespeicherten Zeile fortsetzen</td>
      <td><code class="literal">fit_exponential_checkpointed_csv</code></td>
      <td>Nur für eine lokale Datei auf einem Rechner; verteilte Wiederherstellung<sup id="fnref-distributed-recovery"><a href="#fn-distributed-recovery">4</a></sup> wird nicht unterstützt.</td>
    </tr>
    <tr>
      <td>Eine von sechs Familien an Beobachtungen im Speicher anpassen</td>
      <td><code class="literal">fit</code>, <code class="literal">fit_weibull</code>, <code class="literal">fit_gamma</code> und die übrigen Anpassungen je Familie</td>
      <td>Alle Beobachtungen müssen in den Speicher passen; es gibt keine automatische Modellauswahl.</td>
    </tr>
    <tr>
      <td>Dichte, CDF, Überlebensfunktion, Quantile oder Stichproben für einen Wert oder ein Array berechnen</td>
      <td><code class="literal">logpdf</code>, <code class="literal">cdf</code>, <code class="literal">sf</code>, <code class="literal">ppf</code>, <code class="literal">sample</code></td>
      <td>Sie verwenden eine bereits festgelegte Verteilung und passen keine Parameter an.</td>
    </tr>
    <tr>
      <td>Log-Dichte für einen Wert berechnen</td>
      <td><code class="literal">evaluate_log_density</code></td>
      <td>Passt keine Parameter an.</td>
    </tr>
    <tr>
      <td>Likelihood für mehrere Datenabschnitte berechnen, mit oder ohne Rechtszensierung</td>
      <td><code class="literal">reduce_log_likelihood_chunks</code>, <code class="literal">reduce_lifetime_log_likelihood_chunks</code>, <code class="literal">reduce_value_log_likelihood_chunks</code></td>
      <td>Sie wählen nicht die beste Verteilung aus.</td>
    </tr>
  </tbody>
</table>

Führen Sie für den üblichen Pfad das vollständige Beispiel unten aus:

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
    result = fit_exponential_csv(
        path,
        schema=CsvLifetimeSchema("time", "event_observed"),
        source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
        limits=CsvLifetimeLimits(32768, 32768),
    )

fit = result.fit
assert isinstance(fit, ExponentialFitSuccess)
print(f"rate={fit.rate}; events={fit.event_count}; censored={fit.censored_count}")
```

```text
rate=0.5; events=1; censored=1
```

Dieses Beispiel hat zwei Beobachtungen. In der ersten Beobachtung tritt das Ereignis zum Zeitpunkt 1 ein. In der zweiten Beobachtung wurde bis zum Zeitpunkt 1 kein Ereignis beobachtet; wir wissen also nur, dass die tatsächliche Lebensdauer größer als 1 ist. Damit haben wir ein beobachtetes Ereignis und insgesamt 2 Zeiteinheiten unter Beobachtung. Die geschätzte Rate<sup id="fnref-rate"><a href="#fn-rate">5</a></sup> beträgt `1 / 2 = 0.5` Ereignisse pro Zeiteinheit. Das bedeutet, dass das Modell für vergleichbare Einheiten im Mittel ein halbes Ereignis pro beobachteter Zeiteinheit schätzt. Zur Berechnung einer Ereigniswahrscheinlichkeit muss außerdem ein bestimmtes Zeitintervall festgelegt werden.

## Wie sollte die CSV-Datei aussehen?

`fit_exponential_csv(path, *, schema, source_id, limits)` akzeptiert eine UTF-8-Datei mit zwei Spalten, `time,event_observed`, genau in dieser Reihenfolge. `time` muss eine endliche, nichtnegative Zahl sein. In `event_observed` bedeutet `1`, dass das Ereignis zum aufgezeichneten Zeitpunkt eingetreten ist; `0` bedeutet, dass das Ereignis bis zum aufgezeichneten Zeitpunkt noch nicht beobachtet wurde. Der zweite Fall heißt unabhängige Rechtszensierung<sup id="fnref-right-censoring"><a href="#fn-right-censoring">6</a></sup>.

`CsvLifetimeSchema` benennt die beiden erwarteten Spalten. `PublicSourceId` ist eine öffentliche, nicht geheime Kennung zur Aufzeichnung der Datenprovenienz<sup id="fnref-provenance"><a href="#fn-provenance">7</a></sup>; der lokale Dateipfad wird nicht in das zurückgegebene Ergebnis geschrieben. `CsvLifetimeLimits` legt die maximale Größe jedes Datenabschnitts und die maximale Datenmenge fest, die gleichzeitig in der Verarbeitungsschlange gehalten wird<sup id="fnref-byte-limits"><a href="#fn-byte-limits">8</a></sup>. Beide Werte müssen positiv sein.

Veridist errät keine Spaltennamen, Trennzeichen, Kodierung, fehlenden Daten oder die Bedeutung von Null und Eins. Wenn die Datei dem Format oben nicht entspricht, wird die Verarbeitung mit einer klaren Fehlermeldung beendet; Veridist verändert fragliche Werte nicht automatisch.

## Wie sollten Sie das Ergebnis lesen?

`fit_exponential_csv` gibt immer ein `ExponentialSourceFitResult` zurück. Wenn die Datei erfolgreich verarbeitet wird und die Daten ausreichen, um eine Rate zu schätzen, steht die Anpassung in `result.fit`. Wenn die Datei korrekt verarbeitet wird, aber statistisch keine gültige Rate geschätzt werden kann, zum Beispiel weil die Datei leer ist oder kein Ereignis beobachtet wurde, enthält dasselbe Feld ein typisiertes statistisches Nicht-Schätzergebnis<sup id="fnref-typed-non-estimate"><a href="#fn-typed-non-estimate">9</a></sup>, damit der Grund eindeutig bleibt.

Wenn das Lesen oder Verarbeiten der Datei fehlschlägt, ist `result.fit` gleich `None`. In diesem Fall ist `result.execution` ein typisiertes Ergebnis<sup id="fnref-typed-outcome"><a href="#fn-typed-outcome">10</a></sup>, das Phase und Grund des Fehlers enthält. Prüfen Sie den Ergebnistyp, bevor Sie Modellparameter verwenden.

Das CSV-Modell hält den Lageparameter fest bei null. Die zurückgegebene Anpassung kann die Unsicherheit ihrer Schätzung angeben (siehe den Abschnitt zur Unsicherheit weiter unten), dieser Pfad bietet aber keine Anpassungsgütetests<sup id="fnref-goodness-of-fit"><a href="#fn-goodness-of-fit">12</a></sup>, Gewichte<sup id="fnref-weight"><a href="#fn-weight">13</a></sup>, Kovariaten<sup id="fnref-covariate"><a href="#fn-covariate">14</a></sup>, Datentrunkierung<sup id="fnref-truncation"><a href="#fn-truncation">15</a></sup>, Linkszensierung<sup id="fnref-left-censoring"><a href="#fn-left-censoring">16</a></sup>, Intervallzensierung<sup id="fnref-interval-censoring"><a href="#fn-interval-censoring">17</a></sup>, einen freien Lageparameter<sup id="fnref-free-location"><a href="#fn-free-location">18</a></sup> oder eine automatische Modellauswahl. Die statistische Annahme und die Fälle ohne Schätzung werden in <a href="exponential-right-censoring.md">der Anleitung zur Rechtszensierung</a> erklärt.

## Jede der sechs Familien anpassen

`fit(family, observations)` passt eine von sechs Familien per Maximum-Likelihood an Beobachtungen an, die im Speicher liegen. `family` ist eine `FamilyId` oder ihr Zeichenkettenwert: `exponential`, `weibull_min`, `lognormal`, `gamma`, `normal` oder `gumbel_right`. Jede Familie hat außerdem eine eigene Funktion (`fit_exponential`, `fit_weibull`, `fit_lognormal`, `fit_gamma`, `fit_normal` und `fit_gumbel_right`), die dieselben Beobachtungen und Optionen annimmt.

Die vier Lebensdauerfamilien (`exponential`, `weibull_min`, `lognormal` und `gamma`) nehmen `ExactLifetime`- und `RightCensoredLifetime`-Beobachtungen an. Die zwei Familien auf der gesamten reellen Achse (`normal` und `gumbel_right`) nehmen `ExactValue` und `RightCensoredValue` an, und ein Wert darf negativ sein. Wird das andere Paar übergeben, wird `TypeError` ausgelöst. Jede Familie unterstützt unabhängige Rechtszensierung und `frequency_weights`; `fit_weibull` akzeptiert zusätzlich `fixed_shape`.

```python
from veridist import ExactLifetime, RightCensoredLifetime, fit

times = (120.0, 340.0, 560.0, 800.0, 1250.0, 1700.0)
observations = [ExactLifetime(t) for t in times] + [RightCensoredLifetime(2000.0)]
result = fit("weibull_min", observations)

shape, scale = result.parameters["shape"], result.parameters["scale"]
print(f"{result.family.value} shape={shape:.3f} scale={scale:.1f}")
print(result.observation_count, result.event_count, result.censored_count)
```

```text
weibull_min shape=1.216 scale=1154.2
7 6 1
```

Ein Datenproblem, etwa eine Stichprobe ohne beobachtetes Ereignis, wird als typisierter Fehlschlag zurückgegeben und nicht als Ausnahme ausgelöst. Ein Erfolg stellt `family`, eine schreibgeschützte Zuordnung `parameters` mit den kanonischen Parameternamen, `log_likelihood`, `observation_count`, `event_count`, `censored_count` und `converged` bereit, dazu familienspezifische Attribute wie `rate`, `shape`, `scale`, `mu` oder `sigma`. Ein Fehlschlag stellt `family`, `code` und die Zähler bereit. Die Protokolle `FitSuccess` und `FitFailure` beschreiben diese gemeinsame Oberfläche; vor dem Lesen von Parametern ist daher `isinstance(result, FitSuccess)` die richtige Prüfung. Eine Anpassung meldet nie einen Punkt am Rand ihres Suchbereichs als konvergierte Schätzung; stattdessen gibt sie einen Fehlschlag wie `BOUNDARY_SOLUTION` oder `DEGENERATE_SAMPLE` zurück.

## Verteilungen auf Skalaren und Arrays auswerten

`logpdf`, `cdf`, `sf`, `ppf` und `sample` werten eine Verteilung aus, die Sie bereits festgelegt haben. Die Familie steht zuerst, die Parameter folgen als Schlüsselwörter, zum Beispiel `cdf("gamma", 6.5, shape=5.0, scale=1.0)`. `logpdf` ist die logarithmierte Dichte, `sf` die Überlebensfunktion und `ppf` die Quantilfunktion; `sample(family, size, rng=rng, ...)` zieht mit einem von Ihnen übergebenen numpy-Generator Stichproben aus der Familie.

Der Punkt und jeder Parameter können ein Python- oder numpy-Skalar oder ein Array sein und werden gegeneinander gebroadcastet<sup id="fnref-broadcasting"><a href="#fn-broadcasting">23</a></sup>. Ein skalarer Aufruf gibt ein Python-`float` zurück, jede Array-Eingabe ein `float64`-Array der gebroadcasteten Form. `logpdf` ist außerhalb des Trägers `-inf`, und `ppf` verlangt Wahrscheinlichkeiten, die echt zwischen 0 und 1 liegen. numpy wird nur importiert, wenn Sie ein Array übergeben.

```python
import numpy as np

from veridist import cdf, logpdf, ppf

x = np.array([100.0, 200.0, 400.0])
print(np.round(cdf("weibull_min", x, shape=1.5, scale=500.0), 4))
print(round(logpdf("normal", 0.0, mu=0.0, sigma=1.0), 6))
print(round(ppf("gamma", 0.5, shape=2.0, scale=1.0), 6))
```

```text
[0.0856 0.2235 0.5111]
-0.918939
1.678347
```

Die Familien Exponential, Weibull-Minimum und Rechts-Gumbel verwenden numpy-native Kerne. Die Familien Normal, Lognormal und Gamma werten die verifizierten Skalarkerne Element für Element aus; sie liefern dieselben Werte wie ein skalarer Aufruf, sind aber auf sehr großen Arrays langsam. Die ältere Form, die die Parameter als Zuordnung übergibt, `cdf("gamma", x, {"shape": 2.0, "scale": 1.0})`, funktioniert weiterhin, ist aber veraltet und wird in 3.0 entfernt.

## Beobachtungen aus Arrays erzeugen

`lifetimes_from_arrays(time, event)` erzeugt die Lebensdauerbeobachtungen aus zwei eindimensionalen Spalten gleicher Länge, und `values_from_arrays(value, event)` tut dasselbe für reelle Werte. `event` ist ein boolesches Array oder ein Integer-Array aus 0 und 1; wahr bedeutet, dass das Ereignis beobachtet wurde, und falsch, dass die Beobachtung zu dieser Zeit rechtszensiert ist. Die Spalten werden zuerst als ganze Arrays geprüft, und ein Fehler nennt die erste fehlerhafte Zeile.

```python
import numpy as np

from veridist import fit, lifetimes_from_arrays

time = np.array([120.0, 340.0, 560.0, 800.0, 2000.0])
event = np.array([1, 1, 1, 1, 0])
result = fit("exponential", lifetimes_from_arrays(time, event))
print(round(result.rate, 8))
```

```text
0.00104712
```

## Die Unsicherheit einer Schätzung angeben

Jede erfolgreiche Anpassung hat eine Methode `uncertainty()`; rufen Sie sie auf dem Ergebnis auf, denn sie ist kein Attribut. Sie gibt eine `FitUncertainty` mit den `standard_errors` der Parameter, der Kovarianzmatrix<sup id="fnref-covariance"><a href="#fn-covariance">24</a></sup> `covariance` in der kanonischen Parameterreihenfolge und `confidence_intervals(level=0.95, method="wald")` zurück<sup id="fnref-confidence-interval"><a href="#fn-confidence-interval">11</a></sup>. `method="wald"` bildet Intervalle für positive Parameter auf der logarithmischen Skala und enthält daher nie einen nichtpositiven Wert. `method="profile"` invertiert die Profil-Likelihood<sup id="fnref-profile-likelihood"><a href="#fn-profile-likelihood">25</a></sup>, die bei kleinen Stichproben verlässlicher ist; eine Seite, die nie schneidet, wird als `inf` (oder `0`) gemeldet. `method="exact"` liefert das Chi-Quadrat-Intervall für die Exponentialrate bei unzensierten Daten.

Dasselbe Objekt liefert abgeleitete Größen mit Intervallen: `mean()` (die mittlere Zeit bis zum Ausfall), `quantile(p)` (eine B-Lebensdauer<sup id="fnref-b-life"><a href="#fn-b-life">26</a></sup>; `quantile(0.1)` ist B10) und `survival(t)`. Jede gibt eine `DerivedEstimate` mit `estimate`, `lower`, `upper`, `method` und `level` zurück. Wald-Intervalle verwenden die Delta-Methode auf einer Skala, die den Wertebereich der Größe respektiert, und `method="profile"` steht für die Exponential- und die Weibull-Familie zur Verfügung.

```python
from veridist import ExactLifetime, RightCensoredLifetime, fit

times = (120.0, 340.0, 560.0, 800.0, 1250.0, 1700.0)
observations = [ExactLifetime(t) for t in times] + [RightCensoredLifetime(2000.0)]
uncertainty = fit("weibull_min", observations).uncertainty()

intervals = uncertainty.confidence_intervals()
b10 = uncertainty.quantile(0.1)
print(tuple(round(value, 3) for value in intervals["shape"]))
print(round(b10.estimate, 1), round(b10.lower, 1), round(b10.upper, 1))
```

```text
(0.621, 2.382)
181.4 40.9 804.5
```

Wenn die beobachtete Information singulär ist, ein Weibull-Formparameter festgelegt wurde oder das Ergebnis keine Daten enthält, gibt `uncertainty()` statt einer Ausnahme einen Wert `UncertaintyUnavailable` mit einem stabilen `reason` zurück. Prüfen Sie den Typ, bevor Sie Intervalle lesen.

## Prüfen, wie gut eine Familie zu Ihrer Stichprobe passt

`refit_monte_carlo_gof` in `veridist.inference` prüft, ob eine Stichprobe mit einer der sechs Familien vereinbar ist, mit der Kolmogorov-Smirnov- (`KS`), der Anderson-Darling- (`AD`) oder der Cramér-von-Mises-Statistik (`CVM`). Die Funktion passt die Familie an Ihre Stichprobe an, zieht mit Ihrem numpy-Generator `replicates` gleich große Stichproben aus diesem angepassten Modell, passt jede erneut an und meldet, wie oft die Statistik der erneuten Anpassung die beobachtete erreicht. Die erneute Anpassung jeder Wiederholung hält den p-Wert ehrlich gegenüber den geschätzten Parametern. Die Beobachtungen sind einfache, endliche, exakt beobachtete Zahlen: streng positiv für die Lebensdauerfamilien, beliebige reelle Zahlen für `normal` und `gumbel_right` und mindestens drei für jede Familie außer `exponential`. Zensierte Beobachtungen lösen `TypeError` aus.

Lässt sich die Familie nicht an Ihre Stichprobe anpassen, benennt `GofFitError` den Fehlercode, und es wird kein p-Wert berechnet. Eine Wiederholung, deren erneute Anpassung scheitert, wird in `failed_replicates` gezählt; sie wird weder wiederholt noch verborgen. `assess_families` passt mehrere Familien an dieselbe Stichprobe an, meldet Log-Likelihood, AIC, BIC und p-Wert jeder Familie und wählt die Familie mit dem kleinsten AIC, die nicht verworfen wurde, oder liefert `NONE_ADEQUATE`. Ein Generator steuert die Familien in der angegebenen Reihenfolge, sodass derselbe Seed das Ergebnis reproduziert. Der Aufwand ist für jede bewertete Familie die Zahl der Wiederholungen mal dem Aufwand einer Anpassung.

```python
import numpy as np
from veridist.inference import GofStatistic, assess_families, refit_monte_carlo_gof

hours = [
    140.0, 22.0, 7052.0, 306.0, 390.0, 109.0, 162.0, 246.0, 199.0, 60.0,
    373.0, 310.0, 124.0, 50.0, 40.0, 33.0, 225.0, 88.0, 548.0, 39.0,
]
rng = np.random.default_rng(2024)

test = refit_monte_carlo_gof(
    observations=hours,
    family="lognormal",
    statistics=frozenset({GofStatistic.AD}),
    replicates=199,
    rng=rng,
)
print(test.requested_replicates, test.failed_replicates)
print(test.p_values[GofStatistic.AD] >= 0.05)

result = assess_families(
    observations=hours,
    families=["lognormal", "weibull_min", "gamma", "normal", "exponential"],
    replicates=99,
    rng=rng,
)
for row in result.candidates:
    print(f"{row.family.value} {row.aic:.1f} {row.adequate}")
print(result.selection.code.value, result.selection.selected_family)
```

```text
199 0
True
lognormal 271.4 True
weibull_min 281.1 False
gamma 286.7 False
normal 353.4 False
exponential 292.6 False
SELECTED lognormal
```

Das Bestehen der Angemessenheitsprüfung bedeutet, dass eine Familie bei dieser Schwelle nicht verworfen wurde. Es zeigt nicht, dass die Familie das wahre Modell ist, und eine kleine Stichprobe kann ein schlechtes Modell möglicherweise nicht verwerfen. Hier besteht nur die Lognormalfamilie, und ihr AIC liegt etwa 10 Punkte unter dem der nächsten Familie, sodass sie eindeutig gewählt wird; die anderen vier werden verworfen. Der Kalibrierungsnachweis ist nur eine Simulation mit festem Seed auf einem deklarierten Gitter; den genauen Umfang nennen die <a href="../../KNOWN_LIMITS.de.md">bekannten Grenzen</a>.

## Fortschritt speichern und eine Berechnung fortsetzen

Für Daten, die Ihr Programm bereits in kleine JSON-Abschnitte aufgeteilt hat, importieren Sie `fit_exponential_checkpointed_chunks` aus dem Paket `veridist`. Die Funktion verarbeitet jeden Abschnitt einzeln und speichert die hinreichenden Statistiken<sup id="fnref-sufficient-statistics"><a href="#fn-sufficient-statistics">19</a></sup>, die zum Fortsetzen der Berechnung nötig sind; sie speichert keine Kopie der ursprünglichen Datenzeilen in der Zustandsdatei.

Für CSV-Dateien erledigt `veridist.execution.fit_exponential_checkpointed_csv` dieselbe Arbeit mit Unterstützung für Abbruch. Sie übergeben die Datei, die Spaltendefinition, die Größenbeschränkungen, den Speicherort für den Zustand, die Revisionskennung der Daten und bei Bedarf einen Callback `cancel(cursor)`. Die Revisionskennung ist keine frei wählbare Bezeichnung: Sie muss genau dem aktuellen SHA-256-Hexdigest der CSV-Datei entsprechen, der durch Streaming der Datei berechnet wird, bevor eine Zeile gelesen wird. Wenn der Lauf abgebrochen wird, wird der abgeschlossene Teil zuerst gespeichert; der nächste Lauf kann ab der zuletzt aufgezeichneten Zeile fortsetzen.

Die Datendatei und ihre Revisionskennung müssen zwischen zwei Läufen stabil bleiben: Eine geänderte Datei, eine andere öffentliche Quellkennung oder ein anderes gespeichertes Schema liefert statt einer Fortsetzung eine typisierte Nichtübereinstimmung. Verwenden Sie `veridist.execution.create_checkpointed_csv_store`, um den anfänglichen lokalen Speicher für eine CSV-Datei anzulegen, statt den Checkpoint-Datensatz von Hand zu erstellen. Der Checkpoint-Mechanismus<sup id="fnref-checkpoint"><a href="#fn-checkpoint">20</a></sup> ist dafür gedacht, eine Berechnung auf demselben Rechner fortzusetzen. Die aktuelle Implementierung verwendet lokales SQLite, kopiert aber keine rohen CSV-Zeilen in diesen Speicher. Die Datei wird nicht automatisch verschlüsselt oder authentifiziert; bewahren Sie sie daher wie andere Arbeitsdateien an einem sicheren Ort auf. Das <a href="../../examples/checkpoint_resume.py">Beispiel zum Speichern und Fortsetzen</a> zeigt den Ablauf in zwei Schritten.

## Low-Level-Werkzeuge<sup id="fnref-low-level-tools"><a href="#fn-low-level-tools">21</a></sup> für vorbereitete Daten

Wenn Ihr eigenes Programm die Daten erzeugt oder in Abschnitte teilt, nimmt `IterableDataSource` diese Abschnitte zusammen mit unveränderlichem `DataSourceMetadata` und einer ausdrücklichen `Replayability`-Angabe entgegen. Im Modus `SINGLE_PASS` werden die Daten einmal gelesen. Im Modus `REPLAYABLE` müssen Sie eine Funktion übergeben, die jedes Mal einen neuen Durchlauf über die Daten erzeugt. `CHECKPOINT_REPLAYABLE` ist in diesem Adapter noch nicht implementiert; die Verwendung löst `CHECKPOINT_REQUIRED` aus.

`FAMILY_REGISTRY` und `FamilyId` enthalten die Metadaten der sechs ausgewerteten statistischen Familien. `evaluate_log_density` berechnet die logarithmierte Dichte eines Werts mit den angegebenen Parametern. `reduce_log_likelihood_chunks` summiert dieselbe Berechnung über mehrere Datenblöcke und liefert ein Ergebnis, das nicht davon abhängt, wie die Daten aufgeteilt wurden. Diese Funktionen schätzen keine Modellparameter und ordnen keine Verteilungen. Ihr genauer Vertrag steht in <a href="families-log-density-likelihood.md">Ausgewertete Familien und Log-Likelihood</a>.

Für zensierte Daten addieren `reduce_lifetime_log_likelihood_chunks` (Familien `exponential`, `weibull_min`, `lognormal` und `gamma`, mit `ExactLifetime` und `RightCensoredLifetime`) und `reduce_value_log_likelihood_chunks` (Familien `normal` und `gumbel_right`, mit `ExactValue` und `RightCensoredValue`) die logarithmierte Dichte jeder exakten Beobachtung und die logarithmierte Überlebenswahrscheinlichkeit jeder rechtszensierten. Sie nehmen eine `FamilyId` und die kanonischen Parameter, akkumulieren genau wie `reduce_log_likelihood_chunks` und reproduzieren bei den Parametern einer Anpassung deren `log_likelihood`.

```python
from veridist import (
    ExactLifetime,
    FamilyId,
    RightCensoredLifetime,
    reduce_lifetime_log_likelihood_chunks,
)

chunks = [[ExactLifetime(120.0), ExactLifetime(340.0)], [RightCensoredLifetime(2000.0)]]
result = reduce_lifetime_log_likelihood_chunks(
    FamilyId.WEIBULL_MIN, chunks, shape=1.2, scale=1100.0
)
print(result.observation_count, round(result.total_log_likelihood, 6))
```

```text
3 -16.682975
```

## Was das Paket auf oberster Ebene exportiert

Importieren Sie die öffentliche API aus `veridist`: die sechs Anpassungsfunktionen und `fit`; `FamilyId`, `logpdf`, `cdf`, `sf`, `ppf` und `sample`; die Beobachtungstypen mit `lifetimes_from_arrays` und `values_from_arrays`; die Protokolle `FitSuccess` und `FitFailure`; die CSV- und Checkpoint-Einstiegspunkte, darunter `create_checkpointed_csv_store` und `fit_exponential_checkpointed_csv`; die drei Log-Likelihood-Reducer; und die Fehlerklassen `VeridistError`, `CapabilityError` und `EngineContractError`. Speicher, Puffer, Herkunfts- und Ergebnistypen bleiben in `veridist.engine`, das ebenfalls `CapabilityError` exportiert.

## Umstieg von Version 1.0

Bestehender 1.0-Code funktioniert weiter. Zwei Formen sind veraltet und werden in 3.0 entfernt: die Zuordnungsform von `cdf`, `sf`, `ppf` und `sample` sowie reine `bytes`-Blöcke für `fit_exponential_checkpointed_chunks`. Die <a href="../migration-2.0.md">Migrationsanleitung</a> listet alle Änderungen auf, einschließlich des Lizenzwechsels zur Business Source License 1.1 mit nichtkommerzieller Erlaubnis.

<details>
<summary>Technische Details und Grenzen der aktuellen Version</summary>

Die CSV-Datei wird in einem Durchlauf gelesen. Die Größenbeschränkungen steuern nur das Volumen der Datenabschnitte, die der Adapter selbst hält; sie sind keine Obergrenze für den gesamten Prozessspeicher oder die Ausführungsgeschwindigkeit. Allgemeine Unterstützung für alle Datenquellen, verteilte Ausführung und Wiederherstellung über mehrere Rechner hinweg ist in der aktuellen Version nicht verfügbar; geplante Punkte und genaue Grenzen sind in den <a href="../../KNOWN_LIMITS.de.md">bekannten Grenzen</a> dokumentiert. Eine erfolgreiche Berechnung beweist außerdem nicht, dass die Exponentialverteilung zu den Daten passt.

</details>

## Begriffe auf dieser Seite

<p id="fn-fitting"><strong>1.</strong> <bdi>Distribution fitting</bdi> — Parameter einer Verteilung aus Daten schätzen und ihre Vereinbarkeit mit den Beobachtungen prüfen. <a href="#fnref-fitting" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-scalar"><strong>2.</strong> <bdi>Scalar operation</bdi> — eine Operation auf einem einzelnen Zahlenwert; dieselben Operationen akzeptieren auch Arrays. <a href="#fnref-scalar" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-caller"><strong>3.</strong> <bdi>Caller</bdi> — der Code oder das Programm, das die Bibliotheksfunktion aufruft und ihre Eingaben bereitstellt. <a href="#fnref-caller" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-distributed-recovery"><strong>4.</strong> <bdi>Distributed recovery</bdi> — eine Berechnung mit gemeinsamem Zustand auf einem anderen Rechner oder Dienst fortsetzen. <a href="#fnref-distributed-recovery" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-rate"><strong>5.</strong> <bdi>Rate</bdi> — die erwartete Anzahl von Ereignissen pro Zeiteinheit; eine Rate ist nicht dasselbe wie eine Ereigniswahrscheinlichkeit. <a href="#fnref-rate" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-right-censoring"><strong>6.</strong> <bdi>Independent right censoring</bdi> — das Ereignis wurde bis zum Ende der Beobachtung nicht gesehen, und der Grund für das Beobachtungsende gilt als unabhängig vom Ereigniszeitpunkt. <a href="#fnref-right-censoring" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-provenance"><strong>7.</strong> <bdi>Provenance</bdi> — aufgezeichnete Informationen über Datenquelle und Ausführung, damit das Ergebnis später geprüft werden kann. <a href="#fnref-provenance" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-byte-limits"><strong>8.</strong> <bdi>Byte limits</bdi> — Grenzen dafür, wie viele Daten jeder Abschnitt und die Verarbeitungsschlange gleichzeitig halten dürfen; das ist keine Grenze für den gesamten Arbeitsspeicher des Programms. <a href="#fnref-byte-limits" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-typed-non-estimate"><strong>9.</strong> <bdi>Typed statistical non-estimate</bdi> — ein strukturiertes Ergebnis, das sagt, dass die Berechnung lief, die Daten aber keine gültige Schätzung erlaubten, und den Grund festhält. <a href="#fnref-typed-non-estimate" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-typed-outcome"><strong>10.</strong> <bdi>Typed outcome</bdi> — ein Ergebnis mit definiertem Typ und Code, das Software prüfen kann, ohne freien Text zu analysieren. <a href="#fnref-typed-outcome" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-confidence-interval"><strong>11.</strong> <bdi>Confidence interval</bdi> — ein Intervall, das die Unsicherheit einer Parameterschätzung unter einer festgelegten statistischen Methode zeigt. <a href="#fnref-confidence-interval" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-goodness-of-fit"><strong>12.</strong> <bdi>Goodness-of-fit test</bdi> — eine statistische Prüfung, wie gut die gewählte Verteilungsform zu den Daten passt. <a href="#fnref-goodness-of-fit" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-weight"><strong>13.</strong> <bdi>Weight</bdi> — eine Zahl, die den Beitrag einer Beobachtung zur Berechnung erhöht oder verringert. <a href="#fnref-weight" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-covariate"><strong>14.</strong> <bdi>Covariate</bdi> — eine Variable wie Temperatur oder Druck, die den Ereigniszeitpunkt beeinflussen kann. <a href="#fnref-covariate" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-truncation"><strong>15.</strong> <bdi>Truncation</bdi> — ein Teil der Grundgesamtheit fehlt in der Stichprobe wegen des Eintritts- oder Beobachtungsprozesses, nicht nur weil der Ereigniszeitpunkt unbekannt ist. <a href="#fnref-truncation" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-left-censoring"><strong>16.</strong> <bdi>Left censoring</bdi> — wir wissen nur, dass das Ereignis vor einem bestimmten Zeitpunkt eingetreten ist. <a href="#fnref-left-censoring" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-interval-censoring"><strong>17.</strong> <bdi>Interval censoring</bdi> — wir wissen nur, dass das Ereignis zwischen zwei Zeitpunkten eingetreten ist. <a href="#fnref-interval-censoring" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-free-location"><strong>18.</strong> <bdi>Free location parameter</bdi> — ein Verschiebungsparameter der Verteilung, der aus den Daten geschätzt wird, statt fest vorgegeben zu sein. <a href="#fnref-free-location" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-sufficient-statistics"><strong>19.</strong> <bdi>Sufficient statistics</bdi> — numerische Zusammenfassungen, die zur Parameterschätzung nötig sind und in dieser Berechnung das Speichern aller rohen Zeilen ersetzen. <a href="#fnref-sufficient-statistics" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-checkpoint"><strong>20.</strong> <bdi>Checkpoint</bdi> — gespeicherter Zwischenzustand, der einem kompatiblen Lauf erlaubt, ab dem letzten aufgezeichneten Abschnitt fortzufahren. <a href="#fnref-checkpoint" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-low-level-tools"><strong>21.</strong> <bdi>Low-level tools</bdi> — grundlegendere Funktionen für Fälle, in denen Ihr Programm Datenvorbereitung, Datenaufteilung oder Parameterauswahl selbst erledigt. Diese Werkzeuge sind normalerweise nicht der Schnellstartpfad für Endnutzer. <a href="#fnref-low-level-tools" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-uncertainty"><strong>22.</strong> <bdi>Uncertainty</bdi> — wie weit eine aus einer begrenzten Stichprobe berechnete Schätzung vom wahren Wert entfernt liegen kann, meist als Standardfehler und Konfidenzintervalle angegeben. <a href="#fnref-uncertainty" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-broadcasting"><strong>23.</strong> <bdi>Broadcasting</bdi> — die numpy-Regel, nach der Arrays unterschiedlicher, aber kompatibler Form elementweise kombiniert werden. <a href="#fnref-broadcasting" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-covariance"><strong>24.</strong> <bdi>Covariance matrix</bdi> — eine Tabelle der Varianzen der Parameterschätzungen und der Kovarianzen je zweier Schätzungen; die Wurzeln ihrer Diagonale sind die Standardfehler. <a href="#fnref-covariance" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-profile-likelihood"><strong>25.</strong> <bdi>Profile likelihood</bdi> — die über die übrigen Parameter maximierte Likelihood für jeden Wert des interessierenden Parameters; daraus gebildete Intervalle folgen der tatsächlichen Form der Likelihood. <a href="#fnref-profile-likelihood" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-b-life"><strong>26.</strong> <bdi>B-life</bdi> — die Zeit, bis zu der ein bestimmter Prozentsatz der Einheiten ausgefallen ist; B10 ist die Zeit, bis zu der 10 Prozent ausgefallen sind. <a href="#fnref-b-life" aria-label="Zurück zum Text">↩</a></p>
