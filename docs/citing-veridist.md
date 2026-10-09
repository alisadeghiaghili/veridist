# Citing Veridist

Cite the released version that produced your result. In a report or reproducible
analysis, also state the model family and relevant data assumptions. The
canonical machine-readable record is [`CITATION.cff`](../CITATION.cff).

The examples below cite Veridist 2.1.0. For another release, take the version,
publication date, and release URL from `CITATION.cff`; do not represent an
unversioned repository page as the exact software release.

## DOI

Every release is archived on Zenodo, which assigns two kinds of DOI:

- **Concept DOI** [10.5281/zenodo.23269843](https://doi.org/10.5281/zenodo.23269843). It always resolves to the
  latest release. Use it to cite Veridist as software in general, and it is
  the `doi` recorded in [`CITATION.cff`](../CITATION.cff).
- **Version DOI** [10.5281/zenodo.23269844](https://doi.org/10.5281/zenodo.23269844). It resolves to
  exactly Veridist 2.1.0. Use it when the result depends on that release.

Each later release receives its own version DOI on its Zenodo record. The
examples below keep the release URL; add the DOI that fits your purpose.

## Author identity

The author's published name is **Seyed Ali Sadeghi Aghili**; \`Sadeghi Aghili\`
is treated as the compound family name in the formats below. Research profiles:
[Google Scholar](https://scholar.google.com/citations?user=BDD_JUIAAAAJ&hl=en&authuser=1),
[ResearchGate](https://www.researchgate.net/profile/Seyed-Ali-Sadeghi-Aghili),
and [PeerJ](https://peerj.com/AliSadeghiAghili/).
The author's verified ORCID iD is
[0000-0002-5938-3291](https://orcid.org/0000-0002-5938-3291).

## IEEE

```text
S. A. Sadeghi Aghili, “Veridist,” ver. 2.1.0, Oct. 2026. [Online]. Available:
https://github.com/alisadeghiaghili/veridist/releases/tag/v2.1.0
```

## APA 7

```text
Sadeghi Aghili, S. A. (2026). Veridist (Version 2.1.0) [Computer software].
https://github.com/alisadeghiaghili/veridist/releases/tag/v2.1.0
```

## BibTeX

The `doi` field is the concept DOI. Replace it with the version DOI
`10.5281/zenodo.23269844` to cite exactly 2.1.0.

```bibtex
@software{sadeghi_aghili_veridist_2026,
  author  = {Sadeghi Aghili, Seyed Ali},
  title   = {Veridist},
  version = {2.1.0},
  month   = oct,
  year    = {2026},
  doi     = {10.5281/zenodo.23269843},
  url     = {https://github.com/alisadeghiaghili/veridist/releases/tag/v2.1.0}
}
```

## RIS

```text
TY  - COMP
AU  - Sadeghi Aghili, Seyed Ali
TI  - Veridist
PY  - 2026
DA  - 2026/10/09
VL  - 2.1.0
UR  - https://github.com/alisadeghiaghili/veridist/releases/tag/v2.1.0
ER  -
```

## EndNote XML

```xml
<record>
  <ref-type name="Computer Program">9</ref-type>
  <contributors><authors><author>Sadeghi Aghili, Seyed Ali</author></authors></contributors>
  <titles><title>Veridist</title></titles>
  <dates><year>2026</year></dates>
  <volume>2.1.0</volume>
  <urls><related-urls><url>https://github.com/alisadeghiaghili/veridist/releases/tag/v2.1.0</url></related-urls></urls>
</record>
```

## CSL-JSON

```json
{
  "id": "sadeghi-aghili-veridist-2026",
  "type": "software",
  "title": "Veridist",
  "author": [{"family": "Sadeghi Aghili", "given": "Seyed Ali"}],
  "version": "2.1.0",
  "issued": {"date-parts": [[2026, 10, 9]]},
  "URL": "https://github.com/alisadeghiaghili/veridist/releases/tag/v2.1.0"
}
```

## Chicago author-date

```text
Sadeghi Aghili, Seyed Ali. 2026. Veridist, version 2.1.0. Computer software.
https://github.com/alisadeghiaghili/veridist/releases/tag/v2.1.0.
```

## MLA 9

```text
Sadeghi Aghili, Seyed Ali. Veridist. Version 2.1.0, 9 Oct. 2026,
https://github.com/alisadeghiaghili/veridist/releases/tag/v2.1.0.
```

## Harvard

```text
Sadeghi Aghili, S.A. (2026) Veridist. Version 2.1.0 [Computer software]. Available at:
https://github.com/alisadeghiaghili/veridist/releases/tag/v2.1.0 (Accessed: date).
```

## Vancouver

```text
Sadeghi Aghili SA. Veridist [computer program]. Version 2.1.0. 2026 Oct 9. Available
from: https://github.com/alisadeghiaghili/veridist/releases/tag/v2.1.0
```
