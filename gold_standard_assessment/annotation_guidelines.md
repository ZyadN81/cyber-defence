# Annotation Guidelines

## Purpose

Define consistent manual annotation rules for creating an adjudicated gold subset from weak labels.

## Label space

- Use canonical labels from `data/label_taxonomy.csv`.
- Multi-label assignment is allowed.
- Separate multiple labels using semicolons (`;`).

## Annotation fields

- `annotator_1_labels`: labels from annotator 1.
- `annotator_2_labels`: labels from annotator 2.
- `final_adjudicated_labels`: final post-discussion label set.
- `inclusion_flag`: `INCLUDE`, `EXCLUDE`, or `PENDING`.
- `exclusion_reason`: mandatory when excluded.
- `confidence`: annotator confidence (recommended scale 1-5).
- `adjudication_notes`: rationale for conflict resolution.
- `validation_status`: `NOT_REVIEWED`, `VALIDATED`, `ADJUDICATED`.

## Category definitions

- Malware: malicious code execution, ransomware, malware persistence.
- Network Attacks: phishing, DDoS, network intrusion, active attack operations.
- Data Breach: privacy and data-protection failures, unauthorized data access/exfiltration.
- System Vulnerability: exploitable weaknesses, insecure IoT/system posture, governance gaps.
- Uncategorized: no defensible category assignment from available evidence.

## Borderline-case rules

- If both malware and network-attack evidence exist, keep both labels.
- If only generic cybersecurity language is present, prefer narrower label only when explicit evidence exists.
- If evidence is insufficient for any canonical label, keep label empty and set `inclusion_flag=EXCLUDE` with reason.

## Multi-label handling

- Include all materially supported labels.
- Do not force single-label decisions when text supports multiple threat types.
- Use semicolon-separated normalized labels, sorted alphabetically.

## Uncategorized handling

- Use only when no canonical taxonomy label is defensible.
- Document rationale in `adjudication_notes`.
- Consider queueing these records for taxonomy expansion review.

## Disagreement resolution

1. Annotators label independently.
2. Compare differences and discuss evidence spans.
3. If unresolved, a third reviewer adjudicates.
4. Record final decision in `final_adjudicated_labels` and `adjudication_notes`.

## Overclaim prevention

- Do not call weak ontology-derived labels gold standard.
- Only `final_adjudicated_labels` on validated/adjudicated rows are thesis-gold evidence.
