# benchmarks

Timing and memory measurements of a build. Run each script from the repository root:

```bash
uv run python benchmarks/sink_throughput.py
uv run python benchmarks/streaming_memory.py 200000
```

- `sink_throughput.py [N]` times how fast `InMemorySink` asserts individuals and relations. `N` sets both counts. Without it, the script asserts 100,000 individuals and 50,000 relations.
- `streaming_memory.py [EVENTS] [DIR]` measures the memory a streamed build holds beyond the store, for a low and a high cardinality log. `EVENTS` defaults to 200,000. Pass `DIR` to keep the written files. Without it, they go to a temporary directory. The script reads `/proc/self/statm`, so it runs on Linux only.
