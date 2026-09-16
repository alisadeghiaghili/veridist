<a id="veridist-exponential-right-censoring-de"></a>
# Tutorial zur exponentiellen Rechtszensierung

<a href="exponential-right-censoring.md">Englisch</a> | <a href="exponential-right-censoring.fa.md">Persisch</a> | <a href="exponential-right-censoring.de.md">Deutsch</a>

Dieses Tutorial zeigt den ersten Pfad für Lebensdauerdaten in Veridist: die Anpassung einer Exponentialverteilung<sup id="fnref-fitting"><a href="#fn-fitting">1</a></sup>, wenn einige Beobachtungen rechtszensiert<sup id="fnref-right-censoring"><a href="#fn-right-censoring">2</a></sup> sind. Nutzen Sie ihn, wenn jede Zeile entweder sagt: „Das Ereignis ist zu diesem Zeitpunkt eingetreten“ oder „Bis zu diesem Zeitpunkt wurde das Ereignis noch nicht gesehen“.

## Die Datenidee

Eine Lebensdauerzeile hat zwei Felder:

<table class="docutils" width="100%">
  <thead>
    <tr>
      <th width="30%" align="center"><p align="center">CSV-Feld</p></th>
      <th width="70%" align="center"><p align="center">Bedeutung</p></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code class="literal">time</code></td>
      <td>Eine endliche, nichtnegative beobachtete Zeit.</td>
    </tr>
    <tr>
      <td><code class="literal">event_observed</code></td>
      <td><code class="literal">1</code> bedeutet, dass das Ereignis bei <code class="literal">time</code> eingetreten ist; <code class="literal">0</code> bedeutet, dass das Ereignis bis <code class="literal">time</code> noch nicht eingetreten war.</td>
    </tr>
  </tbody>
</table>

Zum Beispiel bedeutet eine Zeile `1,1`, dass das Ereignis zum Zeitpunkt 1 eingetreten ist. Eine Zeile `1,0` bedeutet, dass die Einheit bis zum Zeitpunkt 1 beobachtet wurde und zu diesem Zeitpunkt noch lebte, funktionierte oder ereignisfrei war. Auch diese zweite Zeile ist nützlich: Sie sagt dem Modell, dass die Lebensdauer länger als 1 ist. Das ist unabhängige Rechtszensierung.

## Beispiel ausführen

Die Schema-Deklaration unten benennt das einzige akzeptierte Kopfzeilenpaar und hält maschinenlesbare Bezeichner von links nach rechts.

```python
from veridist import CsvLifetimeSchema

schema = CsvLifetimeSchema("time", "event_observed")
```

Das vollständige ausführbare Beispiel verwendet dieses Schema mit einer kleinen CSV-Datei.

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

## Ergebnis interpretieren

Das Beispiel hat ein beobachtetes Ereignis und zwei gesamte beobachtete Zeiteinheiten: eine Einheit aus der exakten Ereigniszeile und eine Einheit aus der rechtszensierten Zeile. Die Maximum-Likelihood-Schätzung<sup id="fnref-mle"><a href="#fn-mle">3</a></sup> im Exponentialmodell ist daher:

```{math}
\widehat{rate} = r / \tau = 1 / 2 = 0.5
```

Die Rate<sup id="fnref-rate"><a href="#fn-rate">4</a></sup> wird in Ereignissen pro Zeiteinheit gelesen. Sie ist keine Ereigniswahrscheinlichkeit von 50 Prozent. Wenn Ihre Zeiteinheit Stunden sind, ist die Rate Ereignisse pro Stunde; wenn Ihre Zeiteinheit Tage sind, ist sie Ereignisse pro Tag.

## Was Veridist annimmt

Das Modell nimmt an, dass die Zensierung unabhängig von der Ereigniszeit ist. Einfach gesagt: Der Grund, warum die Beobachtung endet, sollte nicht selbst verborgene Information darüber enthalten, wann das Ereignis eintreten würde. Veridist dokumentiert diese Annahme; es kann sie aber nicht aus der CSV-Datei beweisen.

Der aktuelle öffentliche Pfad schätzt ein Exponentialmodell mit festem Lageparameter aus striktem UTF-8-CSV. Er errät keine Spaltennamen, Trennzeichen, Kodierung, fehlenden Daten oder die Bedeutung von Null und Eins. Wenn die Datei nicht zum erwarteten Vertrag passt, meldet der Adapter das Problem, statt die Daten stillschweigend umzuschreiben.

## Fehlerfälle und aktuelle Grenzen

Eine gültige statistische Schätzung wird nicht zurückgegeben, wenn die Stichprobe leer ist, kein Ereignis beobachtet wurde, ein Ereignis bei gesamter beobachteter Zeit null erscheint oder numerischer Überlauf auftritt. Diese Fälle geben typisierte Nicht-Schätzungen<sup id="fnref-non-estimate"><a href="#fn-non-estimate">5</a></sup> zurück, damit aufrufende Software sie von Dateilesefehlern unterscheiden kann.

Dieser Pfad hat derzeit kein Konfidenzintervall und bietet derzeit keine Konfidenzintervalle<sup id="fnref-confidence-interval"><a href="#fn-confidence-interval">6</a></sup>, Anpassungsgütetests<sup id="fnref-goodness-of-fit"><a href="#fn-goodness-of-fit">7</a></sup>, Trunkierung<sup id="fnref-truncation"><a href="#fn-truncation">8</a></sup>, Linkszensierung<sup id="fnref-left-censoring"><a href="#fn-left-censoring">9</a></sup>, Intervallzensierung<sup id="fnref-interval-censoring"><a href="#fn-interval-censoring">10</a></sup>, Gewichte<sup id="fnref-weight"><a href="#fn-weight">11</a></sup>, Kovariaten<sup id="fnref-covariate"><a href="#fn-covariate">12</a></sup>, einen freien Lageparameter<sup id="fnref-free-location"><a href="#fn-free-location">13</a></sup> oder automatische Modellauswahl.

<details>
<summary>Technische Hinweise zu Skalierung und Evidenz</summary>

Der CSV-Adapter verbraucht einen Iterator-Durchlauf. Sein Byte-Budget für Datenabschnitte ist eine logische Grenze für gehaltene Nutzlast; es ist keine portable Grenze für Prozessspeicher oder RSS und keine Durchsatzbehauptung.

Die historische Evidenz `SCALE-CSV-EXP-01` bewahrt einen Snapshot mit 10k, 100k und 1m Zeilen sowie Abschnittsgrößen von 32 KiB, 64 KiB und 128 KiB nur für diesen strikten Adapter und diesen Schätzer. Das ältere Schema hat nicht genug Provenienz<sup id="fnref-provenance"><a href="#fn-provenance">14</a></sup>, um eine aktuelle Leistungsbehauptung zu stützen. Eine aktuelle Leistungsbehauptung benötigt einen sauberen Lauf, der an die geprüfte Quellrevision und eine dokumentierte Umgebung gebunden ist.

Der historische Snapshot belegt keine allgemeine Unterstützung für große Daten, keinen anderen Adapter, keinen Abbruch, keinen Wiederholungsversuch und kein Checkpointing. Für Speichern und Fortsetzen verwenden Sie den API-Leitfaden und das spezielle Checkpoint-Beispiel.

</details>

## Begriffe in diesem Tutorial

<p id="fn-fitting"><strong>1.</strong> <bdi>Distribution fitting</bdi> — Verteilungsparameter aus Daten schätzen und prüfen, ob die Verteilung eine vertretbare Beschreibung der Beobachtungen ist. <a href="#fnref-fitting" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-right-censoring"><strong>2.</strong> <bdi>Right censoring</bdi> — eine Beobachtung, bei der das Ereignis bis zur letzten beobachteten Zeit nicht eingetreten ist; die exakte Lebensdauer ist also nur als länger als diese Zeit bekannt. <a href="#fnref-right-censoring" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-mle"><strong>3.</strong> <bdi>Maximum-likelihood estimate</bdi> — der Parameterwert, der die beobachteten Daten mit dem gewählten statistischen Modell am besten vereinbar macht. <a href="#fnref-mle" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-rate"><strong>4.</strong> <bdi>Rate</bdi> — die erwartete Anzahl von Ereignissen pro Zeiteinheit; sie ist nicht dasselbe wie eine Ereigniswahrscheinlichkeit. <a href="#fnref-rate" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-non-estimate"><strong>5.</strong> <bdi>Typed non-estimate</bdi> — ein strukturiertes Ergebnis, das sagt, dass die Eingabe verarbeitet wurde, die Daten aber keine gültige statistische Schätzung erlaubten. <a href="#fnref-non-estimate" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-confidence-interval"><strong>6.</strong> <bdi>Confidence interval</bdi> — ein Intervall, das die Unsicherheit um eine Schätzung unter einer bestimmten statistischen Methode ausdrückt. <a href="#fnref-confidence-interval" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-goodness-of-fit"><strong>7.</strong> <bdi>Goodness-of-fit test</bdi> — eine statistische Prüfung, wie gut die gewählte Verteilungsform zu den Daten passt. <a href="#fnref-goodness-of-fit" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-truncation"><strong>8.</strong> <bdi>Truncation</bdi> — ein Stichprobenprozess, bei dem ein Teil der Grundgesamtheit nicht in die beobachtete Stichprobe gelangen kann. <a href="#fnref-truncation" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-left-censoring"><strong>9.</strong> <bdi>Left censoring</bdi> — ein Fall, in dem bekannt ist, dass das Ereignis vor einem bestimmten Zeitpunkt eingetreten ist. <a href="#fnref-left-censoring" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-interval-censoring"><strong>10.</strong> <bdi>Interval censoring</bdi> — ein Fall, in dem bekannt ist, dass das Ereignis zwischen zwei Zeitpunkten eingetreten ist. <a href="#fnref-interval-censoring" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-weight"><strong>11.</strong> <bdi>Weight</bdi> — eine Zahl, die den Beitrag einer Beobachtung zur Berechnung erhöht oder verringert. <a href="#fnref-weight" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-covariate"><strong>12.</strong> <bdi>Covariate</bdi> — eine zusätzliche Variable, etwa Temperatur oder Druck, die helfen kann, Lebensdauer zu erklären. <a href="#fnref-covariate" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-free-location"><strong>13.</strong> <bdi>Free location parameter</bdi> — ein Verschiebungsparameter, der aus Daten geschätzt wird, statt vorher festgelegt zu sein. <a href="#fnref-free-location" aria-label="Zurück zum Text">↩</a></p>
<p id="fn-provenance"><strong>14.</strong> <bdi>Provenance</bdi> — aufgezeichnete Informationen über Quellrevision, Umgebung und Ausführung, damit eine Behauptung später geprüft werden kann. <a href="#fnref-provenance" aria-label="Zurück zum Text">↩</a></p>
