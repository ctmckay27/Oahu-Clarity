# Mari native generative controls — ongoing implementation

This is a diagnostic successor route to causal_v4's insufficient timing/level
adapter. It does not claim completed Mari performance. The selected G1-1 anchor
and profile remain unchanged. CURRENT account state remains authoritative.

The tracked patch adds per-frame, multi-layer numeric activation trajectories to
the pinned CPU Qwen engine and captures native per-frame activations for
calibration. No style prose or emotional presets enter this renderer interface.
Source, model, binary and identity asset checks fail closed.

Build the exact engine revision declared in `build_native.py`, then run
`python -m omega_voice.generative_v5.build_native ENGINE_DIRECTORY`.
The current CLI layout uses ROOT/continuation/engine, ROOT/models/qwen-custom and
the recovered production anchor/profile paths. Model download provenance is
inherited from causal_v4's pinned bootstrap. This diagnostic route is single
request CPU only; it is not qualified for the engine's server or other backends.

Observed 2026-09-19:

* No-control and zero-control patched rendering reproduced carrier WAV bytes.
* A same-speaker punctuation-contrast direction changed native phrase contour
  with zero WER and adequate speaker similarity on one held-out sentence.
* Broader held-out tests rejected that direction: contour was entangled with
  global pitch and vocal force, producing identity drift on new text. The bank
  is retained as negative evidence, not an operative Mari voice asset.
* Qwen2.5-Omni-3B transcribed speech but failed silence and rising/falling delivery
  controls. Its delivery judgments were rejected. An independent evaluator is
  being qualified; no character-specific perceptual pass is claimed.
* 50 native/structural/prior regression tests passed. They do not establish
  acoustic control generalization or voice completion.

Subsequent correction:

* `codec_encode.c` exposes the existing native codec encoder separately from
  the wrapper's Base-only reference/profile creation path. It does not change
  the identity profile. `reference_calibration.py` teacher-forces isolated
  same-waveform contour references and verifies their reconstructed contours.
* Four paired calibration contrasts passed. Their replacement native direction
  passed first free-generation counterfactuals on two unseen texts, with zero
  WER, speaker similarity .657–.814 and steeper falls under stronger finality.
  Perceptual certainty, naturalness and wider performance coverage remain under
  evaluation. Diagnostic derivatives are not production voice assets.
* `interaction.py` stops native PCM delivery and generation on a source-bearing
  interruption. The 0.8-second test delivered exactly "I understand" and did not
  commit the unheard later knowledge event. Full resumed vocal continuity is
  not yet established. The very short interrupted prefix did not pass the
  inherited speaker threshold; do not reclassify that measurement as a pass.
* Runtime optional realized timelines distinguish verified spoken word time
  from actual silence. Existing untimed replay remains backward compatible.
* Native integrity tests reproduce the baseline and zero-control waveform
  byte for byte and reject NaN, truncated and wrong-model packets with no audio.

Next causal correction is paired teacher-forced calibration with content,
duration and identity held fixed and the acoustic contour isolated. This must
pass native free-generation tests, identity, intelligibility, naturalness and
held-out conditions before its control can be admitted. Other performance
channels remain unresolved and are not relabeled as finality.

Diagnostic evidence, calibration banks, dependency manifests and the execution
ledger are saved separately from Git implementation. No production selection
or completion status changes are implied by this checkpoint.
