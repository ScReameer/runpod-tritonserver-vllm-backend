Local test example:

```bash
MODEL="qwen"
curl -s http://localhost:9000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'${MODEL}'",
    "input": "The food was delicious and the waiter...",
    "dimensions": 10,
    "encoding_format": "float"
  }' | jq
```