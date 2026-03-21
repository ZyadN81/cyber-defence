# Gap Analysis

Generated at (UTC): 2026-03-21T20:07:10.014632+00:00

## Current Weaknesses
- Records in weak-label pool: 8592
- Records with unknown labels: 0
- Missing taxonomy labels observed in data: 0
- Taxonomy entries missing tactic mapping: 2
- Manual scenario count: 10

## Why Records Become Uncategorized
- No recognized weak labels for the record.
- Weak labels present but absent from taxonomy.
- Multi-category conflicts resolved by deterministic priority rule (documented in methodology).

## Missing/Ambiguous Label Signals
- No additional missing labels detected in current weak-label extraction.
- Missing tactic mapping in taxonomy: ddos
- Missing tactic mapping in taxonomy: vulnerability

## Thesis-Safe Recommendations
- Treat ontology/rule labels as weak supervision (silver standard), not final gold truth.
- Expand manually validated subset and report inter-annotator agreement before final claims.
- Use scenario metrics as supplementary demonstration, not principal evidence.
- Include mismatch analysis and invalid/skipped row counts in thesis appendix.