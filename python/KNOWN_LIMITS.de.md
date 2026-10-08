# Bekannte Grenzen von Veridist 1.0

Dieses Dokument definiert die Release-Grenze 1.0 für Paketversion `1.0.1`.

## Was diese Grenzen in häufigen Anwendungen bedeuten

- Teams für Zuverlässigkeit, Gesundheit, Kredit, Versicherung, digitale Produkte und Betrieb können ein klar definiertes Time-to-Event-Ergebnis modellieren, wenn die dokumentierten Annahmen gelten. Die aktuellen Modelle passen dieses Ergebnis nicht an Kunden-, Patienten-, Maschinen- oder Umweltmerkmale an.
- Betrugs- und Cybersicherheitsteams können unterstützte skalare Verteilungsberechnungen verwenden, um unter einem bereits festgelegten Referenzmodell ein Signal zu erzeugen. Veridist trainiert keinen Klassifikator, wählt keine Alarmschwelle, verarbeitet keine Feedback-Labels und stellt keinen Adapter für produktive Ereignisströme bereit.
- Teams aus Finanzen, Versicherung, Fertigung und Lieferketten sollten nicht annehmen, dass jede registrierte Familie eine Fit-API hat. Skalare Berechnungen benötigen eine Familie und Parameter, die separat begründet wurden, sofern kein dokumentierter Fit-Pfad existiert.
- In jedem Bereich benötigt die Modellausgabe weiterhin fachliche Validierung, angemessene Stichproben, eine Analyse der Entscheidungskosten und alle erforderlichen rechtlichen, klinischen, sicherheitsbezogenen oder regulatorischen Prüfungen.

- `FIT-CSV-EXP`: Der strikte CSV-Pfad passt nur ein Exponentialmodell mit
  festem Ort und Rate für exakte und unabhängig rechtszensierte Lebensdauern
  an. Weibull-Minimum und Lognormal sind über typisierte Lebensdauerobjekte,
  nicht über eine allgemeine Datei-API, verfügbar. Analytische Gewichte,
  Kovariaten, Trunkierung, Links- und Intervallzensierung sowie freie
  Ortsparameter bleiben nicht unterstützt.
- `CSV-STRICT`: Der mitgelieferte Dateiadapter akzeptiert nur UTF-8-CSV mit
  exakt `time,event_observed`; `1` bezeichnet ein exaktes Ereignis und `0`
  unabhängige Rechtszensierung. Er ist kein allgemeiner CSV- oder
  Tabellenkalkulationsleser. Eine leere Zeile wird nur toleriert, wenn sie
  das Letzte in der Datei ist (zum Beispiel eine abschließende Leerzeile,
  die ein Editor oder eine Tabellenkalkulation angehängt hat); eine leere
  Zeile an anderer Stelle bleibt ein `blank_record`-Fehler.
- `SCALAR-FAMILIES`: Normal-, Gamma-, Weibull-Minimum-, Lognormal-,
  Rechts-Gumbel- und Exponentialfamilien bieten Log-Dichte-, CDF-, Survival-,
  Quantil- und Sampling-Operationen. `logpdf`, `cdf`, `sf` und `ppf` werten
  auch numpy-Arrays aus und führen Broadcasting zwischen dem Punkt und
  arraywertigen Parametern durch; skalare Eingaben liefern weiterhin einen
  Python-`float`. Exponential, Weibull-Minimum und Rechts-Gumbel nutzen
  numpy-native Kerne. Normal, Lognormal und Gamma rufen die verifizierten
  skalaren Kerne Element für Element auf: Die Ergebnisse sind mit dem
  skalaren Pfad identisch, aber bei großen Arrays langsam, weil numpy weder
  `erfc` noch die unvollständige Gammafunktion kennt und Veridist scipy nicht
  als Laufzeitabhängigkeit hat. Inferenz für jede registrierte Familie fehlt.
- `STREAM-SOURCE`: `IterableDataSource` adaptiert Chunk-Iterables des Aufrufers.
  Das Paket enthält keinen Parquet-, Arrow-, Dataframe-, Datenbank- oder
  Netzwerkadapter. Dauerhafte Fortsetzung ist auf den strikten Lebensdauer-CSV-
  Pfad und lokales SQLite auf einem Rechner begrenzt; ein verteilter
  Checkpoint-Store fehlt. Bei `fit_exponential_checkpointed_csv` muss die
  Revisionskennung genau dem aktuellen SHA-256-Hexdigest der CSV-Datei
  entsprechen, geprüft gegen die Datei auf der Festplatte, bevor eine Zeile
  gelesen wird; eine geänderte Datei, eine andere öffentliche Quellkennung
  oder ein anderes gespeichertes Schema liefert statt einer Fortsetzung eine
  typisierte Nichtübereinstimmung. `fit_exponential_checkpointed_chunks`
  sollte mit der Offset-Form `(row_start, payload)` aufgerufen werden, damit
  ein wiederholter Chunk an seinem Zeilenbereich erkannt und übersprungen
  wird, statt ein zweites Mal angewendet zu werden; die veraltete reine
  `bytes`-Form ist deprecated, gibt eine Warnung aus und kann eine
  Wiederholung im Allgemeinen nicht erkennen.
- `INFERENCE-EXP`: Refit-Monte-Carlo-KS/AD/CvM und adequacy-gesteuerte Auswahl
  gelten nur für endliche positive unzensierte Exponentialstichproben. Es gibt
  keine Bootstrap-Auswahlstabilität oder Kalibrierungsbehauptung außerhalb des
  geprüften Gitters.
- `FIT-UNCERTAINTY`: Jeder erfolgreiche Fit liefert `result.uncertainty`: die
  Kovarianz und die Standardfehler aus der beobachteten Information an der
  Schätzung, Wald- und Profil-Likelihood-Konfidenzintervalle (bei
  unzensierten Exponentialdaten zusätzlich das exakte Chi-Quadrat-Intervall)
  sowie Mittelwert, Quantile (B-Lebensdauern) und Überlebenswahrscheinlichkeit
  mit Intervallen. Die Zahlen sind Großstichprobenergebnisse und setzen
  unabhängige Rechtszensierung, eine innere Maximum-Likelihood-Schätzung und
  eine positiv definite Informationsmatrix voraus; Wald-Intervalle brauchen
  zudem eine Stichprobe, die groß genug ist, damit die Likelihood annähernd
  quadratisch ist. Ein Profil-Intervall kann auf einer Seite unbeschränkt sein
  (ausgewiesen als `inf`, bei einem positiven Parameter als `0`, mit einem
  Kennzeichen). Ist die Information singulär oder nicht positiv definit oder
  wurde der Weibull-Formparameter festgelegt, ist `uncertainty` ein
  `UncertaintyUnavailable`-Wert mit einem Grund statt Zahlen. Profil-Intervalle
  für abgeleitete Größen gibt es nur für Exponential- und Weibull-Familie.
  Simulationen mit festem Seed (n = 30 unzensiert und n = 60 mit etwa 30 %
  Rechtszensierung, mindestens 400 Wiederholungen je Familie und Setting)
  ergaben eine Überdeckung der 95-%-Intervalle zwischen etwa 92 % und 96 %;
  außerhalb dieses Gitters wird nichts behauptet. Ein Fit behält seine
  beobachteten Werte (der Exponential-Fit nur seine suffizienten Statistiken),
  damit Intervalle bei Bedarf berechnet werden können.
- `MEMORY-BOUND`: Die Liefergrenze umfasst Nutzdaten in der Warteschlange und
  aktive Verbraucher-Leases bis zur ausdrücklichen Freigabe. Sie ist eine
  logische Grenze für gehaltene Nutzdaten, keine portable RSS-Obergrenze.
- `SCALE-EVIDENCE`: Messungen gelten nur für den exakten Adapter, die Familie,
  Arbeitslast, Plattform, Python-Version, Chunk-Grenze und Kandidaten-SHA. Sie
  belegen weder universellen Durchsatz noch allgemeine Big-Data-Unterstützung
  oder eine breite Out-of-Core-Fähigkeit.
- `LICENSE`: Das Paket verwendet BUSL-1.1; das ist eine quelltextverfügbare,
  aber keine Open-Source-Lizenz. Die zusätzliche Nutzungserlaubnis in `LICENSE`
  gestattet die produktive Nutzung nur für nichtkommerzielle Zwecke (private
  Nutzung, akademische Forschung und Lehre sowie nichtkommerzielle Aktivitäten
  gemeinnütziger Organisationen); jede andere produktive Nutzung, auch die
  interne betriebliche Nutzung, erfordert eine kommerzielle Lizenz des
  Lizenzgebers. Die Lizenz wechselt am 2030-09-05 zu Apache License, Version
  2.0 (Apache-2.0).
- `SOURCE-MUTATION-STAT`: Der Mutationsstatus `VERIFIED_UNCHANGED` einer
  CSV-Ausführung vergleicht die vom Betriebssystem gemeldete Identität der
  Datei (Gerät, Inode, Größe und Änderungszeit) vor und nach dem Lesen. Das
  ist kein Inhalts-Hash und erkennt nicht jedes Überschreiben an Ort und
  Stelle, das diese vier Werte unverändert lässt.
- `CONTEXT-REDACTION`: Die Redaktion des Fehlerkontexts ist eine feste, nach
  `_` aufgeteilte Zulassungsliste von Schlüsselnamen; sie verwirft Schlüssel,
  deren Teile einer verbotenen Liste entsprechen, prüft aber nie die Werte.
  Ein Schlüssel, der diese Teile zufällig vermeidet (zum Beispiel `filepath`
  statt `file_path`), wird nicht erkannt; dies ist daher keine allgemeine
  Datenredaktion. Der Ausnahmetext zeigt Zahlen und kurze, codeartige Token
  aus dem Kontext und ersetzt jede andere Zeichenkette, etwa einen Pfad oder
  eine URI, durch `<redacted>`; das Kontext-Mapping selbst bleibt unverändert.

[English](KNOWN_LIMITS.md) | [فارسی](KNOWN_LIMITS.fa.md)
