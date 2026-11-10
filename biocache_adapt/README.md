# CacheGen BioCache endpoint/edge proxy

The package implements `reset`, `prefill_archive`, `query`,
`persistent_state`, `static_extra_state`, and `temporary_state`.

The endpoint encodes K/V, transfers the byte stream under a configurable
bandwidth, and the edge restores dense K/V only while answering. Codec
compression/recovery timings and network timing are separate. The encoded
edge archive is persistent; client payload and restored dense K/V are
temporary.

```bash
python -m biocache_adapt --dry-run --budget 25% \
  --output /tmp/cachegen.jsonl --manifest /tmp/cachegen-manifest.json
```

Actual-byte 12.5%, 25%, and 50% budgets and the unified
`biocache-proxy-v1` needle/length-scan schema are supported. The official
engine is delayed-imported by `make_cachegen_engine`.
