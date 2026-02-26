### Run 

uv run uvicorn app:app --reload --port 8000

### docker 

docker build -t deployment-project .
docker run -p 8000:8000 deployment-project

### Test

curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "NASA launched a new rocket"}'