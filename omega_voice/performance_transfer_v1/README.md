# MARI_PERFORMANCE_TRANSFER_v1

A controlled cross-speaker performance-conditioning experiment for Mari.

## Question

Can the same frozen Mari Voice v1 / G1-1 speaker identity sound materially more believable when the renderer receives a real human conversational performance as a separate acoustic reference?

## Separation law

speaker identity = Mari Voice v1 x-vector

performance evidence = external human reference clip

The donor is never Mari identity authority. A reference may condition performance only when its exact waveform, transcript, source URL, license, attribution and donor identifier are bound into the render receipt.

## v1 scope

- whole-utterance generation only;
- x-vector profile only;
- one human performance reference per render;
- no native trajectory;
- no incremental text release;
- no prose acting instruction;
- no random humanization;
- no expressive-graft speaker substitution.

This isolation is intentional. V1 asks whether the renderer can absorb useful high-dimensional performance information at all before attempting continuous conversational integration.

## Reference bank

The production workflow builds a deterministic three-donor reference bank from spontaneous human speech in the AMI Meeting Corpus (CC BY 4.0). Donors are performance/naturalness evidence only.

## Evaluation

For each target line:
A = Mari x-vector, no performance reference.
B = same Mari x-vector + human performance reference.

Text, seed, model and Mari speaker profile are held fixed. Outputs are gated for intelligibility and Mari-anchor speaker similarity. Improvement in acting/conversational believability remains a human listening claim.
