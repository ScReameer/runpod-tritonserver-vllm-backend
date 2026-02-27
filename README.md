Local test:

```bash
docker compose up --build
```

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

Test after deploy:
```bash
MODEL="qwen"
RUNPOD_API_KEY="YOUR_RUNPOD_API_KEY"
curl -s https://<ENDPOINT_ID>.api.runpod.ai/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -d '{
    "model": "'${MODEL}'",
    "input": "The food was delicious and the waiter...",
    "dimensions": 10,
    "encoding_format": "float"
  }' | jq
```