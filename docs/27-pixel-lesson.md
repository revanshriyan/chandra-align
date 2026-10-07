# The 27-pixel lesson — a methods note

*Background writing for the CHANDRA-ALIGN evidence track. Everything below
is grounded in measured cases from the failure gallery.*

## The thesis

A registration pipeline has two jobs, and they are not the same job:

1. Find a geometric fit consistent with the image data.
2. Decide whether that fit is *true*.

Every failure in our gallery is a case where job 1 succeeded and job 2 was
the only thing standing between us and a false claim.

## Why inlier RMSE lies

RANSAC minimizes disagreement *among the inliers it chose*. That is a
measure of self-consistency, not of truth. The Phase 11 `sun_flip` case is
the cleanest demonstration we have: 21 inliers, inlier RMSE 0.87 px —
sub-pixel by any in-sample standard — and 485.79 px from exact synthetic
truth. Polarity reversal built a coherent false world, and the inliers all
agreed to live in it.

This is not a rare pathology. Any symmetric or repetitive structure can do
it; illumination change does it routinely on the Moon. The inlier RMSE is
the fit admiring itself in a mirror.

## Why gates must fail closed

The Phase 13 TMC-2 window `tmc2_fore_nadir_w0006` had 1,141 unique inliers
and was rejected: residual 2.5755 px against a 2.50 px gate. A thousand
witnesses, overruled by a ruler. That is the gate doing its job — because
the alternative is a pipeline that accepts whatever is popular. Twenty-one
of forty-two TMC-2 windows failed the same way, and every one is in the
published table, because a gate that only ever says yes is decoration.

Fail-closed is not pessimism; it is the only posture compatible with
job 2. A rejected true registration costs a row in a table. An accepted
false one costs the credibility of every row around it.

## Why independent verification exists

If the fit cannot grade itself, something else must. Phase 10's pixel
area-check re-derives agreement from raw image correlation in 36
independent cells, using a criterion (sharp correlation peak at near-zero
offset) that shares no machinery with the feature matcher. On true
transforms it verifies 36/36 cells; on a 60 px-wrong transform, 0/36 —
see `docs/images/trust-map-sample.svg`.

The Phase 14 phase-congruency case shows why this layer is not optional:
the gate passed nominally (8 inliers, 1.22 px), but the inliers clustered
on one edge and the area-check found 0/36 cells verified. Two independent
instruments disagreed with the fit's self-assessment. We reported
"promising but unverified" — the only honest sentence available.

## Why negative results are load-bearing

The MINIMA experiment failed to beat the baseline and was published
anyway, because two different matchers converging on the same geometry
(~1.2° rotation, ~0.99 scale, ~108 px Y shift) is corroboration no
single-matcher success can provide. The RIFT2 zero-correspondence record
stays in the open for the same reason: a lab notebook with the failed
experiments torn out is fiction.

## The practice, in one paragraph

Measure the pipeline, never the pipeline's opinion of itself. Freeze the
gates before the data arrives. Report every window, including the ones
that embarrass you. Hash the evidence so nobody — including future you —
can quietly revise it. And when the numbers disagree with the story you
wanted, publish the numbers.
