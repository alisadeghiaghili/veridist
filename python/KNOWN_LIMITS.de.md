# Bekannte Grenzen von Veridist 2.1

Dieses Dokument definiert die Release-Grenze 2.1 für Paketversion `2.1.0`.

## Was diese Grenzen in häufigen Anwendungen bedeuten

- Teams für Zuverlässigkeit, Gesundheit, Kredit, Versicherung, digitale Produkte und Betrieb können ein klar definiertes Time-to-Event-Ergebnis modellieren, wenn die dokumentierten Annahmen gelten. Die aktuellen Modelle passen dieses Ergebnis nicht an Kunden-, Patienten-, Maschinen- oder Umweltmerkmale an.
- Betrugs- und Cybersicherheitsteams können unterstützte skalare Verteilungsberechnungen verwenden, um unter einem bereits festgelegten Referenzmodell ein Signal zu erzeugen. Veridist trainiert keinen Klassifikator, wählt keine Alarmschwelle, verarbeitet keine Feedback-Labels und stellt keinen Adapter für produktive Ereignisströme bereit.
- Teams aus Finanzen, Versicherung, Fertigung und Lieferketten können jede der sechs registrierten Familien an ihre Beobachtungen anpassen. Eine Familie muss weiterhin separat begründet werden: Anpassung und Unsicherheitsangabe ordnen keine Familien, und Anpassungsgütetests sowie angemessenheitsgesteuerte Modellauswahl decken nur endliche, unzensierte Stichproben ab (`INFERENCE-GOF`).
- In jedem Bereich benötigt die Modellausgabe weiterhin fachliche Validierung, angemessene Stichproben, eine Analyse der Entscheidungskosten und alle erforderlichen rechtlichen, klinischen, sicherheitsbezogenen oder regulatorischen Prüfungen.

- `FIT-CSV-EXP`: Der strikte CSV-Pfad passt nur ein Exponentialmodell mit
  festem Ort und Rate für exakte und unabhängig rechtszensierte Lebensdauern
  an. Die übrigen Familien (Weibull-Minimum, Lognormal, Gamma, Normal und Rechts-Gumbel) werden über typisierte Beobachtungen im Speicher mit `fit` und den Funktionen je Familie angepasst, nicht über eine Datei-API; `frequency_weights` werden dort unterstützt. Analytische Gewichte,
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
  fügt je Familie einen solchen Lauf hinzu). Das Bestehen der
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
- `SCALE-EVIDENCE`: Messungen gelten nur für den exakten Adapter, die Familie,
  Arbeitslast, Plattform, Python-Version, Chunk-Grenze und Kandidaten-SHA. Sie
  belegen weder universellen Durchsatz noch allgemeine Big-Data-Unterstützung
  oder eine breite Out-of-Core-Fähigkeit. Die erhaltenen Messungen für 2.0.0
  liegen im Verzeichnis `python/evidence/` des Quell-Repositorys (sie sind nicht
  Teil der Distributionsarchive). Sie wurden am 2026-10-09 am Kandidaten-Commit
  `19ecf1062978fe0f894625e65b5a113ba1b68166` mit einem Messworker auf einem
  GitHub-Actions-Linux-Runner (CPython 3.11.17) und einem Windows-Runner
  (CPython 3.11.9) erstellt und werden von
  `tools/check_scale_csv_exponential_evidence.py` und
  `tools/check_log_likelihood_scale_evidence.py` akzeptiert. Bei 1.000.000
  Zeilen dauerte die strikte exponentielle CSV-Anpassung über die Chunk-Grenzen
  32, 64 und 128 KiB unter Linux 14,1 bis 18,9 Sekunden und unter Windows 14,8
  bis 19,8 Sekunden; der Spitzenwert des Prozessspeichers (RSS) lag bei 40,2 bis
  45,0 MiB unter Linux und 40,7 MiB unter Windows. Der Reduzierer für den
  exakten Log-Likelihood-Zustand benötigte für eine Million Beobachtungen je nach
  Familie 3,2 bis 9,2 Sekunden unter Linux und 3,3 bis 10,6 Sekunden unter
  Windows (die Lognormal-Familie ist am langsamsten), bei einem RSS-Spitzenwert
  von 25,8 MiB unter Linux und 25,3 MiB unter Windows. Das sind beschreibende
  Zeitmessungen dieser Läufe, keine Garantien.
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
