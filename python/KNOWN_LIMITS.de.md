# Bekannte Grenzen von Veridist 2.2

Dieses Dokument definiert die Release-Grenze 2.2 für Paketversion `2.2.0`.

## Was diese Grenzen in häufigen Anwendungen bedeuten

- Teams für Zuverlässigkeit, Gesundheit, Kredit, Versicherung, digitale Produkte und Betrieb können ein klar definiertes Time-to-Event-Ergebnis modellieren, wenn die dokumentierten Annahmen gelten. Die aktuellen Modelle passen dieses Ergebnis nicht an Kunden-, Patienten-, Maschinen- oder Umweltmerkmale an.
- Betrugs- und Cybersicherheitsteams können unterstützte skalare Verteilungsberechnungen verwenden, um unter einem bereits festgelegten Referenzmodell ein Signal zu erzeugen. Veridist trainiert keinen Klassifikator, wählt keine Alarmschwelle, verarbeitet keine Feedback-Labels und stellt keinen Adapter für produktive Ereignisströme bereit.
- Teams aus Finanzen, Versicherung, Fertigung und Lieferketten können jede der sechs registrierten Familien an ihre Beobachtungen anpassen. Eine Familie muss weiterhin separat begründet werden: Anpassung und Unsicherheitsangabe ordnen keine Familien, und Anpassungsgütetests sowie angemessenheitsgesteuerte Modellauswahl decken nur endliche, unzensierte Stichproben ab (`INFERENCE-GOF`).
- In jedem Bereich benötigt die Modellausgabe weiterhin fachliche Validierung, angemessene Stichproben, eine Analyse der Entscheidungskosten und alle erforderlichen rechtlichen, klinischen, sicherheitsbezogenen oder regulatorischen Prüfungen.

- `FIT-CSV-EXP`: Der strikte CSV-Pfad passt nur ein Exponentialmodell mit
  festem Ort und Rate für exakte und unabhängig rechtszensierte Lebensdauern
  an. Die übrigen Familien (Weibull-Minimum, Lognormal, Gamma, Normal und Rechts-Gumbel) werden über typisierte Beobachtungen im Speicher mit `fit` und den Funktionen je Familie angepasst, nicht über den CSV-Pfad; für exakte Daten, die nicht in den Speicher passen, passt `veridist.scale` alle sechs Familien an (`SCALE-ENGINE`). `frequency_weights` werden nur im Speicher unterstützt. Analytische Gewichte,
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
  als Laufzeitabhängigkeit hat. Anpassungsgütetests und Modellauswahl werden unter `INFERENCE-GOF` beschrieben.
- `STREAM-SOURCE`: `IterableDataSource` adaptiert Chunk-Iterables des Aufrufers.
  Abgesehen vom CSV-Adapter enthält das Paket keinen Netzwerkadapter; die
  Quellen von `veridist.scale` stehen unter `SCALE-ENGINE`. Dauerhafte
  Fortsetzung ist auf den strikten Lebensdauer-CSV-Pfad und die
  Partitions-Checkpoints von `veridist.scale` begrenzt, beide auf lokalem
  SQLite auf einem Rechner; ein verteilter
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
- `INFERENCE-GOF`: Der Refit-Monte-Carlo-Anpassungsgütetest mit KS/AD/CvM
  (`refit_monte_carlo_gof`) und die angemessenheitsgesteuerte Auswahl zwischen
  Familien (`assess_families`) decken alle sechs registrierten Familien ab, für
  endliche, exakt beobachtete Stichproben im Arbeitsspeicher: streng positive
  Werte für die Lebensdauerfamilien, beliebige reelle Werte für die Normal-
  und die Rechts-Gumbel-Familie und mindestens drei Beobachtungen für jede
  Familie außer der Exponentialfamilie, die eine benötigt. Zensierte
  Beobachtungen werden mit `TypeError` abgelehnt; Anpassungsgütetests für
  zensierte Daten gibt es nicht. Lässt sich die beobachtete Stichprobe nicht
  anpassen, benennt `GofFitError` den Fehlercode, und es wird kein p-Wert
  berechnet; eine Wiederholung, deren Neuanpassung scheitert, wird in
  `failed_replicates` gezählt und nicht erneut versucht. Ein p-Wert ist eine
  Monte-Carlo-Schätzung mit binomialem Standardfehler, und der Aufwand ist die
  Zahl der Wiederholungen mal dem Aufwand einer Anpassung (eine Bewertung
  fügt je Familie einen solchen Lauf hinzu). Beide akzeptieren standardmäßig
  höchstens 100000 Beobachtungen: Eine größere Stichprobe löst
  `RefitGofRefused` (Code `SCALE_REFIT_REFUSED`) aus, bevor etwas angepasst
  oder gezogen wird, und das Limit lässt sich mit dem Schlüsselwortargument
  `max_observations` anheben. Das Bestehen der
  Angemessenheitsprüfung bedeutet nur, dass eine Familie nicht verworfen
  wurde, nicht dass sie das wahre Modell ist, und das AIC ordnet nur die
  verglichenen Familien. Der Kalibrierungsnachweis ist eine Simulation mit
  festem Seed auf einem deklarierten Gitter (Stichprobenumfänge 30 und 100, für
  jede nicht exponentielle Familie ein oder zwei Parameterwerte, nominelles
  Niveau 0,10); außerhalb davon wird nichts behauptet. Eine
  Bootstrap-Auswahlstabilität gibt es nicht.
- `FIT-UNCERTAINTY`: Jeder erfolgreiche Fit liefert `result.uncertainty()`: die
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
- `LL-CENSORED`: `reduce_lifetime_log_likelihood_chunks` und
  `reduce_value_log_likelihood_chunks` unterstützen nur unabhängige
  Rechtszensierung. Die Log-Survival-Terme zensierter Beobachtungen sind in
  ihren Tail-Entwicklungen auf etwa `1e-12` relativ genau und werden von der
  Log-Dichte-Orakelhülle nicht abgedeckt. Ein Term, den binary64 nicht
  darstellen kann, oder ein so großer Gamma-Formparameter (etwa `1e5` und mehr),
  dass die Entwicklung der unvollständigen Gammafunktion nicht konvergiert, ist
  ein typisierter Fehlschlag und kein geratener Wert.
- `DEPRECATED-FORMS`: Die Zuordnungsform von `cdf`, `sf` und `ppf`, die Form
  `sample(family, size, parameters, rng)` und reine `bytes`-Chunks für
  `fit_exponential_checkpointed_chunks` funktionieren weiterhin, lösen
  `DeprecationWarning` aus und werden in 3.0 entfernt.
- `MEMORY-BOUND`: Die Liefergrenze umfasst Nutzdaten in der Warteschlange und
  aktive Verbraucher-Leases bis zur ausdrücklichen Freigabe. Sie ist eine
  logische Grenze für gehaltene Nutzdaten, keine portable RSS-Obergrenze.
- `SCALE-ENGINE`: `veridist.scale` passt die sechs registrierten Familien an
  exakte, unzensierte, endliche Beobachtungen an, die in begrenzten Stapeln
  gelesen werden. Rechtszensierung, Häufigkeits- und analytische Gewichte,
  Trunkierung, Kovariaten und freie Ortsparameter werden im großen Maßstab nicht
  unterstützt; verwenden Sie für zensierte Daten die Anpassungen im Speicher.
  Die Quellen sind numpy-Arrays, Parquet-Dateien, Arrow-Tabellen, -Reader und
  -Streams (`veridist[arrow]`), Polars-Frames (`veridist[polars]`) und
  DuckDB-Relationen (`veridist[duckdb]`); eine eigene Quellklasse gehört nicht
  zur öffentlichen API. Nur Quellen, deren Partitionen einen Fingerabdruck haben
  (`NumpySource` und `ParquetSource`), können mit Checkpoint fortgesetzt werden;
  eine Arrow-, Polars- oder DuckDB-Quelle ist nicht fortsetzbar
  (`SOURCE_REVISION_UNAVAILABLE`), und ein Reader oder Stream kann nur einmal
  gelesen werden, sodass eine Mehrdurchlauf-Familie ihn ohne Spool ablehnt
  (`SPOOL_REQUIRED`). `weibull_min` und `gumbel_right` lesen die Quelle
  mindestens 3- und höchstens 17-mal, innerhalb eines `pass_budget` von
  standardmäßig 8; Profil-Likelihood-Intervalle, nur für diese beiden Familien,
  lesen mindestens 2 und höchstens 9 weitere Durchläufe innerhalb ihres eigenen
  Budgets (standardmäßig 6); der Anpassungsgüte-Bericht für großes n ist ein
  Durchlauf und eine Einschließung des Kolmogorov-Smirnov-Abstands, nie ein
  p-Wert. Checkpoints sind lokales SQLite auf einem Host und an die numerische
  Umgebung gebunden; es gibt keinen verteilten Executor. Ein Stapel enthält
  höchstens `max_rows` Werte (standardmäßig 262144), und ein Zustand wächst
  nicht mit der Zeilenzahl; das ist eine logische Schranke, keine RSS-Grenze.
  Innerhalb einer numerischen Umgebung sind die Ergebnisse bitgleich für jede
  Stapelgröße, jedes Partitionslayout, jeden Executor, jede Workerzahl, jeden
  Spool und jeden Fortsetzungspunkt; über Plattformen, Python-Builds und
  numpy-Versionen hinweg wird nur eine Übereinstimmung im Rahmen des
  Rundungsfehlers von `log`, `exp` und `pow` erwartet, und es wird keine
  numerische Toleranz erklärt. Erhaltene Messungen dieser Engine sind unter
  `SCALE-EVIDENCE` beschrieben, und die Refit-Monte-Carlo-Tests von
  `INFERENCE-GOF` lehnen jede Scale-Quelle ab.
- `SCALE-EVIDENCE`: Messungen gelten nur für den exakten Adapter, die Familie,
  Arbeitslast, Plattform, Python-Version, Chunk-Grenze und Kandidaten-SHA. Sie
  belegen weder universellen Durchsatz noch allgemeine Big-Data-Unterstützung
  oder eine breite Out-of-Core-Fähigkeit. Die erhaltenen Messungen für 2.2.0
  liegen im Verzeichnis `python/evidence/` des Quell-Repositorys (sie sind nicht
  Teil der Distributionsarchive). Sie wurden am 2026-10-21 am Kandidaten-Commit
  `e316d125e0ae434beaf0c69ebeee3238c1076f6b` mit einem Messworker auf einem
  GitHub-Actions-Linux-Runner (CPython 3.11.17) und einem Windows-Runner
  (CPython 3.11.9), beide mit 4 vCPUs, erstellt und werden von
  `tools/check_scale_engine_evidence.py`,
  `tools/check_scale_csv_exponential_evidence.py` und
  `tools/check_log_likelihood_scale_evidence.py` akzeptiert. Bei 1.000.000
  Zeilen dauerte die strikte exponentielle CSV-Anpassung über die Chunk-Grenzen
  32, 64 und 128 KiB unter Linux 14,4 bis 19,7 Sekunden und unter Windows 15,2
  bis 21,2 Sekunden; der Spitzenwert des Prozessspeichers (RSS) lag bei 40,2 bis
  43,8 MiB unter Linux und 41,3 MiB unter Windows. Der Reduzierer für den
  exakten Log-Likelihood-Zustand benötigte für eine Million Beobachtungen je
  nach Familie 3,1 bis 9,2 Sekunden unter Linux und 3,4 bis 11,3 Sekunden unter
  Windows (die Lognormal-Familie ist am langsamsten), bei einem RSS-Spitzenwert
  von 25,98 MiB unter Linux und 26,25 MiB unter Windows. Für die Scale-Engine
  (`veridist.scale`), die exponentielle Beobachtungen in Parquet-Dateien mit
  `max_rows` 262144 und einem `pass_budget` von 8 angepasst hat, umfasst die
  Leiter auf beiden Runnern 1.000.000, 10.000.000 und 100.000.000 Zeilen und nur
  unter Linux 1.000.000.000 Zeilen, dort nur für die vier
  Einzeldurchlauf-Familien. Bei 100.000.000 Zeilen benötigten die
  Einzeldurchlauf-Familien unter Linux 1,3 bis 5,4 Sekunden mit dem seriellen
  Executor und 1,8 bis 3,7 Sekunden mit vier Prozessen, `weibull_min` 57,9 und
  26,4 Sekunden in 6 Durchläufen und `gumbel_right` 39,8 und 18,7 Sekunden in 4
  Durchläufen; unter Windows benötigten die Einzeldurchlauf-Familien seriell 1,9
  bis 7,6 Sekunden und mit vier Prozessen 1,8 bis 4,5 Sekunden, `weibull_min`
  83,1 und 45,6 Sekunden und `gumbel_right` 59,4 und 32,8 Sekunden. Bei
  1.000.000.000 Zeilen benötigten die Einzeldurchlauf-Familien unter Linux
  seriell 13,1 bis 57,5 Sekunden und mit vier Prozessen 12,3 bis 30,1 Sekunden.
  Der RSS-Spitzenwert des Messprozesses (die Worker sind nicht enthalten) betrug
  in jeder Linux-Zelle bei 100.000.000 und 1.000.000.000 Zeilen 221,6 MiB und
  unter Windows bei 100.000.000 Zeilen 78,4 bis 97,8 MiB (seriell) und 56,1 bis
  59,7 MiB (vier Prozesse). Für jede Familie und jede Zeilenzahl waren die
  Digests von Ergebnis, Zustand und Durchläufen über den seriellen Executor,
  vier Prozesse, ein zweites Partitionslayout und einen unterbrochenen und
  fortgesetzten Lauf hinweg identisch. Das sind beschreibende Zeitmessungen
  dieser Läufe, keine Garantien.
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
