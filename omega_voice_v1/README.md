# Omega Voice / Mari Voice v1

Status: **PRODUCTION READY** for ordinary English speaking voice.

## Stable call

After one-time runtime bootstrap:

```bash
bash production/bootstrap_runtime.sh
python production/omega_voice.py --text "Give me the facts." --performance neutral --output mari.wav
```

Performance states: `neutral`, `dry_amusement`, `controlled_irritation`, `reassuring`, `focused_urgency`.

The interface fails closed if the exact engine/model/profile is unavailable. It never silently substitutes another voice.

## Production identity

- Anchor candidate: `G1-1`
- Anchor SHA-256: `73bb894fc75e18dddf76e3327f3bf48e161b7dd8b30fbe4949c3ac7318c29567`
- Persistent profile: 8 KB Qwen x-vector
- Profile SHA-256: `9c49146ae2bd3c5a32c686ad1705fd912391af4a39db0d63e9dc353640d304fa`
- Production model: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- Model revision: `0c0e3051f131929182e2c023b9537f8b1c68adfe`
- Engine revision: `e391ec5467b0218eeb175f4888ad65b259d1e7c7`
- Precision: BF16/unquantized

## Verification

- 6/6 unseen neutral sentences passed.
- 4/4 performance states passed identity and quality; all 4 were acoustically distinct.
- 2/2 long-form passages passed.
- Mean UTMOS: 4.539; worst: 4.501.
- Mean ECAPA anchor similarity: 0.756; threshold: 0.601.
- All ASR checks passed; no clipping.
- Independent clean-runner cold start: PASS.
- Missing-profile fail-closed test: PASS; no fallback waveform created.

Automatic perceptual metrics are screening/verification evidence, not fabricated human listening scores. The final v1 acoustic selection was made by ChatGPT under Carl's explicit delegated Mari Voice v1 acoustic authority, using prior rejection evidence and the executed tests above.

Singing, multilingual continuity, and radically transformed acoustic embodiments are outside the closed v1 English-speaking scope.