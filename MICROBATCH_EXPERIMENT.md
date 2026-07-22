# Irodori stream microbatch experiment

`/v1/tts/stream` can optionally coalesce the next chunk from concurrent requests and run
those chunks through one GPU batch. The default remains disabled, so existing deployments
keep the previous single-request locking behavior.

## Environment variables

- `IRODORI_STREAM_MICROBATCH_ENABLED=false`: enable the experimental stream path.
- `IRODORI_STREAM_MICROBATCH_MAX_CHUNKS=8`: maximum compatible chunks per GPU batch.
- `IRODORI_STREAM_MICROBATCH_WINDOW_MS=30`: maximum collection window for a batch.
- `IRODORI_STREAM_MICROBATCH_QUEUE_MAX_CHUNKS=64`: pending queue capacity.
- `IRODORI_STREAM_MICROBATCH_RESULT_TIMEOUT_SECONDS=180`: per-chunk result timeout.

Compatible requests must use the same runtime, LoRA adapter, duration settings, sampling
settings, and decode settings. Text, captions, seeds, and reference voices may differ.
Automatic duration is predicted for every item and the batch samples up to the longest item.

## Safe rollout and rollback

Build `Dockerfile.runpod.microbatch-overlay`, deploy it to a separate endpoint, and enable the
feature only there. Do not change the production template during validation.

The implementation is isolated in the commit after the baseline commit `eb893bb`. Revert that
implementation commit, or redeploy the existing `20260706-watchdog-lb` image, to restore the
previous behavior.
