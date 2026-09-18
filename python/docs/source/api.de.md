<a id="veridist-api-de"></a>
# Veridist-API-Leitfaden

<a href="api.md">Englisch</a> | <a href="api.fa.md">Persisch</a> | <a href="api.de.md">Deutsch</a>

Dieser Leitfaden erklärt die aktuelle öffentliche API in Veridist. Für die Anpassung einer Exponentialverteilung<sup id="fnref-fitting"><a href="#fn-fitting">1</a></sup> an Lebensdauerdaten ist der übliche Weg, eine CSV-Datei an die Hauptfunktion zu übergeben und das zurückgegebene Ergebnis zu lesen. Wenn eine Berechnung lange dauert und unterbrochen werden kann, können Sie den Fortschritt speichern und später fortsetzen. Skalarwerkzeuge<sup id="fnref-scalar"><a href="#fn-scalar">2</a></sup> und Datenströme, die vom Aufrufer<sup id="fnref-caller"><a href="#fn-caller">3</a></sup> verwaltet werden, stehen ebenfalls für technischere Anwendungsfälle bereit.

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
      <td>Log-Dichte für einen Wert berechnen</td>
      <td><code class="literal">evaluate_log_density</code></td>
      <td>Passt keine Parameter an.</td>
    </tr>
    <tr>
      <td>Likelihood für mehrere Datenabschnitte berechnen</td>
      <td><code class="literal">reduce_log_likelihood_chunks</code></td>
      <td>Wählt nicht die beste Verteilung aus.</td>
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

Das aktuelle Modell hält den Lageparameter bei null fest. Dieser Pfad bietet noch keine Konfidenzintervalle<sup id="fnref-confidence-interval"><a href="#fn-confidence-interval">11</a></sup>, Anpassungsgütetests<sup id="fnref-goodness-of-fit"><a href="#fn-goodness-of-fit">12</a></sup>, Gewichte<sup id="fnref-weight"><a href="#fn-weight">13</a></sup>, Kovariaten<sup id="fnref-covariate"><a href="#fn-covariate">14</a></sup>, Datentrunkierung<sup id="fnref-truncation"><a href="#fn-truncation">15</a></sup>, Linkszensierung<sup id="fnref-left-censoring"><a href="#fn-left-censoring">16</a></sup>, Intervallzensierung<sup id="fnref-interval-censoring"><a href="#fn-interval-censoring">17</a></sup>, freien Lageparameter<sup id="fnref-free-location"><a href="#fn-free-location">18</a></sup> oder automatische Modellauswahl. Die statistische Annahme und Nicht-Schätzfälle werden im <a href="exponential-right-censoring.de.md">Tutorial zur Rechtszensierung</a> erklärt.

## Fortschritt speichern und eine Berechnung fortsetzen

Für Daten, die Ihr Programm bereits in kleine JSON-Abschnitte aufgeteilt hat, importieren Sie `fit_exponential_checkpointed_chunks` aus dem Paket `veridist`. Die Funktion verarbeitet jeden Abschnitt einzeln und speichert die hinreichenden Statistiken<sup id="fnref-sufficient-statistics"><a href="#fn-sufficient-statistics">19</a></sup>, die zum Fortsetzen der Berechnung nötig sind; sie speichert keine Kopie der ursprünglichen Datenzeilen in der Zustandsdatei.

Für CSV-Dateien erledigt `veridist.execution.fit_exponential_checkpointed_csv` dieselbe Arbeit mit Unterstützung für Abbruch. Sie übergeben die Datei, die Spaltendefinition, die Größenbeschränkungen, den Speicherort für den Zustand, die Revisionskennung der Daten und bei Bedarf einen Callback `cancel(cursor)`. Wenn der Lauf abgebrochen wird, wird der abgeschlossene Teil zuerst gespeichert; der nächste Lauf kann ab der zuletzt aufgezeichneten Zeile fortsetzen.

Die Datendatei und ihre Revisionskennung müssen zwischen zwei Läufen stabil bleiben. Der Checkpoint-Mechanismus<sup id="fnref-checkpoint"><a href="#fn-checkpoint">20</a></sup> ist dafür gedacht, eine Berechnung auf demselben Rechner fortzusetzen. Die aktuelle Implementierung verwendet lokales SQLite, kopiert aber keine rohen CSV-Zeilen in diesen Speicher. Die Datei wird nicht automatisch verschlüsselt oder authentifiziert; bewahren Sie sie daher wie andere Arbeitsdateien an einem sicheren Ort auf. Das <a href="../../examples/checkpoint_resume.py">Beispiel zum Speichern und Fortsetzen</a> zeigt den Ablauf in zwei Schritten.

## Low-Level-Werkzeuge<sup id="fnref-low-level-tools"><a href="#fn-low-level-tools">21</a></sup> für vorbereitete Daten

Wenn Ihr eigenes Programm die Daten erzeugt oder in Abschnitte teilt, nimmt `IterableDataSource` diese Abschnitte zusammen mit unveränderlichem `DataSourceMetadata` und einer ausdrücklichen `Replayability`-Angabe entgegen. Im Modus `SINGLE_PASS` werden die Daten einmal gelesen. Im Modus `REPLAYABLE` müssen Sie eine Funktion übergeben, die jedes Mal einen neuen Durchlauf über die Daten erzeugt. `CHECKPOINT_REPLAYABLE` ist in diesem Adapter noch nicht implementiert; die Verwendung löst `CHECKPOINT_REQUIRED` aus.

`FAMILY_REGISTRY` und `FamilyId` enthalten die Metadaten der fünf geprüften statistischen Familien. `evaluate_log_density` berechnet die Log-Dichte eines Werts mit den übergebenen Parametern. `reduce_log_likelihood_chunks` summiert dieselbe Berechnung über mehrere Datenabschnitte und gibt ein Ergebnis zurück, das nicht davon abhängt, wie die Daten aufgeteilt wurden. Diese Funktionen schätzen keine Modellparameter, ordnen keine Verteilungen und erstellen keine Likelihoods für zensierte Daten. Ihr genauer Vertrag steht in <a href="families-log-density-likelihood.md">geprüfte Familien und Log-Likelihood</a>.

<details>
<summary>Technische Details und Grenzen der aktuellen Version</summary>

Die CSV-Datei wird in einem Durchlauf gelesen. Die Größenbeschränkungen steuern nur das Volumen der Datenabschnitte, die der Adapter selbst hält; sie sind keine Obergrenze für den gesamten Prozessspeicher oder die Ausführungsgeschwindigkeit. Allgemeine Unterstützung für alle Datenquellen, verteilte Ausführung und Wiederherstellung über mehrere Rechner hinweg ist in der aktuellen Version nicht verfügbar; geplante Punkte und genaue Grenzen sind in den <a href="../../KNOWN_LIMITS.de.md">bekannten Grenzen</a> dokumentiert. Eine erfolgreiche Berechnung beweist außerdem nicht, dass die Exponentialverteilung zu den Daten passt.

</details>

## Begriffe auf dieser Seite

<p id="fn-fitting"><strong>1.</strong> <bdi>Distribution fitting</bdi> — Parameter einer Verteilung aus Daten schätzen und ihre Vereinbarkeit mit den Beobachtungen prüfen. <a href="#fnref-fitting" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-scalar"><strong>2.</strong> <bdi>Scalar operation</bdi> — eine Operation, die jeweils mit einem numerischen Wert arbeitet, nicht mit einem ganzen Array. <a href="#fnref-scalar" aria-label="Zurück zum Text">↩</a></p>
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
