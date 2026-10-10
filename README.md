# EDSSearch

Prerequisites:

## .NET 10 SDK
- Install the .NET 10 SDK from https://dotnet.microsoft.com/download/dotnet/10.0
- Verify: `dotnet --version`

## Redis / RedisSearch (Vector Database)
The app stores its search index and document vectors in **Redis**, using the **RediSearch** module
(via the `NRedisStack` client). RediSearch provides the vector similarity search (KNN with FLAT index,
inner-product metric) and full-text search.

The Redis connection string is configured in `appsettings.json` -> `Eds:Redis`.

### Redis must have the RediSearch module loaded
RediSearch is not built into vanilla Redis. The Redis server must be started with the module:

**Docker (recommended):**
```powershell
docker run -d --name redis -p 6379:6379 redis/redis-stack:7-alpine
```
(`redis/redis-stack` images bundle RediSearch; plain `redis:7-alpine` does NOT have it.)

**Redis for Windows:**
Download a Redis build that includes RediSearch, e.g. from
https://github.com/RedisLabs/redis-search/releases, and start it with:
```
redis-server.exe --load-module RedisSearch.dll
```

Verify the module is loaded:
```powershell
redis-cli module list
# Should include: { "name": "search", "version": ... }
```

### Localhost (development)
Run a local Redis with RediSearch (see Docker command above), then set in `appsettings.json`:
```json
"Redis": "localhost:6379"
```

Verify Redis is reachable:
```powershell
redis-cli ping
# Should return: PONG
```

### Azure (production)
Use **Azure Cache for Redis** (supports RediSearch on the Premium tier and above):
- Create an Azure Cache for Redis instance from the Azure portal.
- Get the hostname and access key from the portal (Settings -> Access keys).
- Enable "Non-TLS/SSL port" if you want to connect without TLS, or use the TLS endpoint (port 6380).

Connection string formats:
- **With TLS (recommended):** `yourcache.redis.cache.windows.net:6380,ssl=true,password=YOUR_ACCESS_KEY`
- **Without TLS:** `yourcache.redis.cache.windows.net:6379,ssl=false,accesskey=YOUR_ACCESS_KEY`

Update `appsettings.json`:
```json
"Redis": "yourcache.redis.cache.windows.net:6379,ssl=true,accesskey=YOUR_ACCESS_KEY"
```

### Index details (what the app creates)
On startup the app runs `FT.CREATE eds:idx` with:
- A `FLAT` vector index on field `v` (`TYPE FLOAT32`, `DIM` from the embedding model, `DISTANCE_METRIC IP`)
- Text fields: `path`, `title`, `keywords`, `description`, `content` (with field weights)
- Key prefix `eds:doc:` so document hashes are stored as `eds:doc:/en/about`

Embedding model files (multilingual-e5-small)

Put these two files in this folder (names/paths are set in appsettings.json -> Eds:Embedding):

Save as	Download URL	Size
model.onnx	https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx	~470 MB (fp32)
sentencepiece.bpe.model	https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/sentencepiece.bpe.model	~5 MB

Windows PowerShell (add -Proxy if you are behind the corporate proxy):

New-Item -ItemType Directory -Force models | Out-Null
$p = @{ Proxy = "http://your-proxy:8080"; ProxyUseDefaultCredentials = $true }   # remove if not needed
Invoke-WebRequest @p -Uri "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx" -OutFile models\model.onnx
Invoke-WebRequest @p -Uri "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/sentencepiece.bpe.model" -OutFile models\sentencepiece.bpe.model

curl:

curl -L -o models/model.onnx https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx
curl -L -o models/sentencepiece.bpe.model https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/sentencepiece.bpe.model

Large files redirect to a Hugging Face CDN host, so the firewall must allow huggingface.co and its CDN (.hf.co / cdn-lfs.huggingface.co). If blocked, download on another machine and copy the files over.

Smaller int8 alternative (~120 MB, slightly lower quality): file model_opt2_QInt8.onnx from https://huggingface.co/nixiesearch/multilingual-e5-small-onnx (the model card lists it; check the Files tab for the exact path). Save it as model.onnx and keep the same sentencepiece.bpe.model.

Or export it yourself (Python + optimum):

pip install optimum[onnxruntime]
optimum-cli export onnx --model intfloat/multilingual-e5-small models/

Other XLM-R based models work the same way (multilingual-e5-base, bge-m3, ...): replace the files and reindex. The embedding size is detected from the model automatically. e5 models need the "query: " / "passage: " prefixes (already the defaults); bge-m3 does not - set both prefixes to "".

Check it works after dotnet run:

GET /api/embed-test?text=Hello world

It returns the token ids, dimension (384 for e5-small) and vector norm (should be 1).

Local LLM for the summary (optional, Eds:Llm in appsettings.json)

The summary box writes a short new answer from the best passages of the top results, with [1] [2] citations (like an AI overview). It needs a small instruct model in ONNX Runtime GenAI format: a folder that contains genai_config.json + model.onnx (+ .data) + tokenizer files. Put the folder at models/llm (or change Eds:Llm:ModelPath). If the folder is missing the app still starts and falls back to the extractive summary (quoted sentences) - the log says "LLM summary disabled".

Install the package once: dotnet add package Microsoft.ML.OnnxRuntimeGenAI (if you get a native-library / version error, install it after Microsoft.ML.OnnxRuntime and let NuGet pick matching versions).

Option A - Qwen2.5 Instruct (recommended for Thai; Qwen2.5 lists Thai among its supported languages)

Build the ONNX folder yourself with the official builder (needs Python, downloads the original weights from Hugging Face once):

pip install onnxruntime-genai torch transformers
python -m onnxruntime_genai.models.builder -m Qwen/Qwen2.5-1.5B-Instruct -o models/llm -p int4 -e cpu
Qwen2.5-1.5B-Instruct: about 1 GB after int4, faster on CPU, weaker answers.
Qwen2.5-3B-Instruct: about 2 GB, slower, noticeably better Thai.
Keep "Template": "chatml" in appsettings.json.
Option B - Phi-3.5-mini (ready-made ONNX from Microsoft, no Python build)
hf download microsoft/Phi-3.5-mini-instruct-onnx --include "cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4/*" --local-dir models/phi35

Set "ModelPath": "models/phi35/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4" and "Template": "phi3". (Check the exact folder name on the repo's Files tab.) Its Thai is weaker than Qwen2.5.

Notes
CPU speed is roughly 5-15 tokens/s depending on model and cores; the answer streams in while it is written.
Only one answer is generated at a time; a new search cancels the previous one.
The prompt tells the model to use only the numbered sources and not to invent figures, but small models can still make mistakes - the page shows a "please verify" note under AI answers.