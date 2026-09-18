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
  Tabellenkalkulationsleser.
- `SCALAR-FAMILIES`: Normal-, Gamma-, Weibull-Minimum-, Lognormal- und
  Rechts-Gumbel-Familien bieten skalare Log-Dichte-, CDF-, Survival-, Quantil-
  und Sampling-Operationen. Arrays, eine einheitliche Fit-API und Inferenz für
  jede registrierte Familie fehlen.
- `STREAM-SOURCE`: `IterableDataSource` adaptiert Chunk-Iterables des Aufrufers.
  Das Paket enthält keinen Parquet-, Arrow-, Dataframe-, Datenbank- oder
  Netzwerkadapter. Dauerhafte Fortsetzung ist auf den strikten Lebensdauer-CSV-
  Pfad und lokales SQLite begrenzt; ein verteilter Checkpoint-Store fehlt.
- `INFERENCE-EXP`: Refit-Monte-Carlo-KS/AD/CvM und adequacy-gesteuerte Auswahl
  gelten nur für endliche positive unzensierte Exponentialstichproben. Es gibt
  keine Bootstrap-Auswahlstabilität oder Kalibrierungsbehauptung außerhalb des
  geprüften Gitters.
- `MEMORY-BOUND`: Die Liefergrenze umfasst Nutzdaten in der Warteschlange und
  aktive Verbraucher-Leases bis zur ausdrücklichen Freigabe. Sie ist eine
  logische Grenze für gehaltene Nutzdaten, keine portable RSS-Obergrenze.
- `SCALE-EVIDENCE`: Messungen gelten nur für den exakten Adapter, die Familie,
  Arbeitslast, Plattform, Python-Version, Chunk-Grenze und Kandidaten-SHA. Sie
  belegen weder universellen Durchsatz noch allgemeine Big-Data-Unterstützung
  oder eine breite Out-of-Core-Fähigkeit.
- `LICENSE`: Das Paket verwendet BUSL-1.1 mit der in `LICENSE` beschriebenen
  zusätzlichen Apache-2.0-Nutzungserlaubnis und wechselt am 2030-09-05 zu
  Apache-2.0.

[English](KNOWN_LIMITS.md) | [فارسی](KNOWN_LIMITS.fa.md)
