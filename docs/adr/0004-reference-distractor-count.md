# Reference distractor count: 2, uniform across the grid; 0 at distance=0

The eval grid sweeps hop_count (1–5) and distance ({0, 3, 9, 21, 45}) at a fixed reference distractor count, so distractor interference doesn't confound the hop×distance comparison. But distractor capacity (`max_distractor_capacity`) varies by cell — it's 0 at distance=0, and its minimum across all other cells is 2 (at hop_count=1, distance=3) versus a maximum of 90 (at hop_count=5, distance=45).

Two options were considered for the reference count:

1. **A fixed absolute count (2) uniform across every distance>=3 cell**, forced to 0 at distance=0.
2. **A count that scales with per-cell capacity** (e.g. a fixed fraction of capacity), so distractor *density* stays comparable across cells instead of a fixed count becoming negligible at large distance.

Option 2 was rejected because it reintroduces exactly the coupling that filler tokens exist to eliminate: distance and distractor count are meant to be independently sweepable axes (fillers pad the distance gap regardless of how many distractors occupy it), and making distractor_count a function of capacity — which is itself a function of distance — would make the "fixed reference distractor count" covary with distance again.

**Decision: option 1.** Reference distractor_count = 2 for every distance>=3 cell in the main hop×distance grid, and 0 at distance=0 (forced, since capacity there is 0 — this is the distractor-free floor cell described in ADR 0003's distance=0 rationale, not an inconsistency).

**Known trade-off, by design:** the ratio of distractor density to available capacity shrinks sharply at large distance (2 of 90 possible slots at hop_count=5, distance=45). The main grid therefore measures degradation under constant *absolute* interference, not constant *relative* interference. Measuring how degradation responds to distractor density scaling with available space is a legitimate, different question — that's what the separate 1D distractor-count ablation (swept at a single fixed distance, independent of the hop×distance grid) exists to answer, rather than conflating it into this grid.
