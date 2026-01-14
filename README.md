# DeepSeek R1 on AMD MI350X GPU

## Docker Configuration

| Property | Value |
|----------|-------|
| Container Name | `deepseek-r1` |
| Image | `rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210` |
| Model | `deepseek-ai/DeepSeek-R1` |
| API Port | `8000` |
| Network Mode | Host |
| Max Model Length | 8192 tokens |

## Hardware

- **GPU:** AMD MI350X
- **Tensor Parallelism:** 8 workers (TP0-TP7)
- **ROCm Version:** 7.0.0

## Performance Benchmarks

Tested on 2025-12-25:

| Test | Completion Tokens | Time | Speed |
|------|-------------------|------|-------|
| Short | 100 | 1.50s | 66.79 tokens/s |
| Medium | 300 | 4.30s | 69.77 tokens/s |
| Long | 500 | 6.89s | 72.53 tokens/s |

**Average generation speed: ~70 tokens/s**

## API Usage

### Check available models
```bash
curl http://localhost:8000/v1/models
```

### Chat completion request
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Docker Commands

### View container status
```bash
docker ps | grep deepseek
```

### View logs
```bash
docker logs deepseek-r1 --tail 50
```

### Check throughput stats
```bash
docker logs deepseek-r1 2>&1 | grep "throughput" | tail -5
```

## Related Services

| Container | Image | Purpose |
|-----------|-------|---------|
| `open-webui` | `ghcr.io/open-webui/open-webui:main` | Web UI for LLM interaction |
| `deepseek-r1` | `rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210` | DeepSeek R1 model server |
