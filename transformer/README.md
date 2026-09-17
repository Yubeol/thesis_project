# Transformer 1차 논문 초안 학습

이 모듈은 **제공된 근거로 짧은 영문 1차 초안**을 만드는 역할만 맡습니다.
검색, 외부 LLM 호출, 번역, 최종 논문 작성, DB 변경은 수행하지 않습니다.

설계상 위치: 질문 → LLM Input Analyzer → 1차 Hybrid RAG → **Transformer** →
LLM Draft/Gap Analyzer → 2차 RAG → LLM Finalizer.
현재 저장소의 Agent는 제목/주제 번역 → RAG → Transformer Adapter → Finalizer 구조입니다.
이번 구현은 새로운 분석기나 2차 검색을 기존 Agent에 임의로 추가하지 않습니다.

## 입력과 팀원 연결

```python
from transformer.inference import generate_draft

draft = generate_draft(
    title="Digital participation in music fandoms",
    topic="Online fandom participation",
    research_question="How does online participation support community activity?",
    paper_evidence=["Participants shared translations and coordinated community activities online."],
    news_evidence=[],  # 없으면 [NO_NEWS_EVIDENCE]
    instruction="Write a concise academic first draft based on the provided evidence.",
    model_path="artifacts/transformer_model",
    device="auto",  # cpu / cuda / auto
)
print(draft)
```

반환형은 `str`이며 `Introduction:`, `Body:`, `Conclusion:` 순서입니다.
근거 없이 만든 수치, 인용 형태, 근거에서 찾을 수 없는 대문자 단어(기관/인명 후보)를 발견하면
해당 절을 `Insufficient evidence to generate this section reliably.`로 바꿉니다.
모델이 절을 생성하지 못한 경우도 동일한 문구를 사용하고 경고를 남깁니다.
이는 보수적인 휴리스틱이므로 정상 문장을 거부할 수 있으며 사실 검증을 완전히 보장하지 않습니다.
저자·기관·날짜·수치를 만들지 말라는 지시문은 학습과 추론에 같은 방식으로 들어갑니다.
최종 인용 조립과 사실 검증은 근거를 가진 Finalizer에서 수행해야 합니다.

`DraftGenerator(model_path, device).generate(...)`도 사용할 수 있습니다.
인스턴스의 `last_diagnostics`에서 누락·거부된 절을 확인할 수 있습니다.
모델 다운로드는 학습 시에만 필요하고, 저장된 모델의 추론은 `local_files_only=True`입니다.

기존 `agent/draft_generator/transformer_adapter.py`의
`generate_draft(title=..., topic=..., evidence=...)` 호출도 지원합니다.
이전 combined evidence는 paper evidence 한 항목으로 전달됩니다.
뉴스를 별도로 구분하거나 research question을 전달하려면 위의 신규 인자를 사용하세요.
기존 서비스의 mock 기본값은 유지합니다. 평가한 **full 모델**을 준비한 후 서비스 실행 환경에서
`TRANSFORMER_USE_MOCK=false`, `TRANSFORMER_MODEL_PATH=<모델 폴더 절대 경로>`를 설정하세요.
이 두 환경변수는 이번 작업에서 자동 변경하지 않습니다.

## 파일 역할

| 파일 | 역할 |
|---|---|
| `configs/config.py` | 모델명·길이·학습 파라미터의 공통 기본값 |
| `configs/requirements-runtime.txt` | torch 설치와 분리한 학습/추론 의존성 |
| `configs/requirements-dataset.txt` | DB export 전용 의존성 |
| `dataset/build_dataset.py` | 읽기 전용 PostgreSQL 조회, 품질 선별, 중복 그룹화, 분할, JSONL |
| `preprocessing/prompts.py` | 입력 검증·공통 지시문·필드별 토큰 예산·출력 절 처리 |
| `training/data.py` | JSONL 검증·논문 ID 누수 검사·파일 해시·토큰화 |
| `training/runtime.py` | CPU/CUDA 선택과 실제 CUDA 커널 forward/backward 진단 |
| `training/train.py` | LoRA 학습·loss·체크포인트·모델 병합 저장·dry-run gate |
| `training/setup-windows.ps1` | Windows Python 3.12 환경 준비·CUDA 검사·기존 torch 보존 |
| `inference/generate.py`, `inference/__init__.py` | CLI와 재사용 가능한 `generate_draft()` |
| `../tests/transformer/` | 분할·중복·누수·DB read-only·학습 gate·토큰 예산·호환성 테스트 |
| `../.github/workflows/train-transformer.yml` | EC2 export와 Windows GPU 학습 연결 |

패키지별 `__init__.py`는 역할별 모듈을 import할 수 있게 합니다.
기존 파일 변경은 학습 workflow와 루트 `.gitignore`이며, 기존 RAG/LLM/FastAPI/DB 코드는 변경하지 않습니다.

신규 파일은 위 표의 구현 파일들과 이 README를 포함해 20개입니다.
패키지 파일은 `transformer/__init__.py`, `configs/__init__.py`, `dataset/__init__.py`,
`preprocessing/__init__.py`, `training/__init__.py`, `inference/__init__.py` 6개입니다.
테스트 파일은 `tests/transformer/test_dataset.py`(export·중복·분할),
`test_training_contract.py`(gate·GPU 요구·추론 호환·workflow),
`test_tokenization.py`(입력/출력 토큰 예산)입니다.

## 이번 구현에서 실제 확인한 결과 (2026-09-17)

| 검증 | 결과 |
|---|---|
| PostgreSQL 읽기 전용 조회 | 논문 187편, 그중 세 절이 NULL이 아닌 논문 144편 |
| 완결 문장·영어·절 중복 검사 후 | 131편, train 105 / validation 13 / test 13 |
| 제외 | 짧은 완결 문장/절 부족 49, 영어 외/미상 6, 중복 절 1 |
| 학습/검증 토큰 예산 | 입력 최대 268/256 tokens, target 128 초과 0건 |
| Transformer 테스트 | 14개 통과 |
| 기존 pipeline/agent/backend 테스트 | 기존 `.venv`에서 76개 통과 |
| 최신 원격 `d8f3253` 호환성 | 별도 복사본에 이번 모듈을 얹어 90개 통과, ML 의존성 테스트 3개 skip; 그 3개는 위 학습 환경의 14개 테스트에 포함되어 통과 |
| Windows setup 스크립트 | Windows PowerShell에서 기존 torch 유지·CUDA 검사 통과 |
| GPU dry-run | GTX 1050, cu126, 2 optimizer steps, 저장 및 재로딩 성공 |
| CPU dry-run | 같은 코드에서 CPU forward/backward·저장·추론 성공 |
| GPU full | 3 epochs, 42 steps, validation loss 0.7687 → 0.6086 |
| full 실행 시간 | 학습 루프 약 71.7초; CPU 시험도 함께 실행한 당시의 관측치 |
| GPU peak tensor allocation | 367 MiB; CUDA context/캐시/다른 앱까지 합친 전체 VRAM 사용량은 아님 |
| CLI·Python 함수·기존 Agent Adapter | 저장한 모델로 실제 호출 성공 |
| GitHub-hosted dispatch/Windows 서비스 실행 | 아직 실행하지 않음. Runner 라벨·서비스 권한은 GitHub에서 확인 필요 |

로컬 검증 모델은 `artifacts/transformer_model/`, 최종 코드의 GPU dry-run은
`artifacts/transformer_gpu_verified/`, CPU dry-run은 `artifacts/transformer_cpu_verified/`에 있습니다.
위 경로의 metadata 및 로그에 실제 설정과 데이터 해시가 있습니다.
이 로컬 데이터/모델/가상환경은 `.gitignore`로 제외되며 Git에 업로드하지 않습니다.

**생성 품질은 아직 합격으로 판정하지 않았습니다.**
Full 모델의 validation 예제에서는 세 절 구조가 나오지 않았고 근거 없는 명칭도 탐지되어
반환문이 근거 부족 문구로 바뀌었습니다. 별도의 짧은 CLI 예제에서는 근거에 있는 본문 문장은
나왔지만 서론·결론이 누락됐습니다. 헤더 보정은 모델이 세 절을 학습했다는 증거가 아닙니다.
현재 결과는 학습/저장/연결 기능의 검증이며, 서비스 적용 전 검수된 정답과 생성 품질 평가가 필요합니다.
Held-out test 13편은 아직 생성 품질 평가에 사용하지 않았습니다.
기존 Agent의 mock 설정이나 배포 상태는 변경하지 않았습니다.

작업 종료 시 원격 main이 로컬보다 3커밋 앞서 있었습니다. 원격 변경은
`ingest-papers.yml` 및 백엔드 입력 검증/테스트이며 이번 변경 파일과 겹치지 않습니다.
최신 코드는 별도 스냅샷에서 검증했고, 현재 작업 브랜치에 merge하거나 commit/push하지는 않았습니다.
기존의 미추적 `docs/architecture/entity-resolution.md`도 그대로 유지했습니다.

## 설치 및 GPU 확인 (Windows PowerShell, 저장소 루트에서 실행)

```powershell
python --version
nvidia-smi
powershell -NoProfile -ExecutionPolicy Bypass -File transformer/training/setup-windows.ps1
& .venv-transformer/Scripts/python.exe -m transformer.training.runtime --device cuda
& .venv-transformer/Scripts/python.exe -m pip install -r transformer/configs/requirements-dataset.txt
```

`python`은 Python **3.12**여야 합니다. 기존 `.venv`가 3.14인 환경을 확인했으므로
그 환경은 교체하지 않고 `.venv-transformer`를 사용합니다.
새 CUDA 환경은 기존 프로젝트의 핀에 맞춰 `torch==2.8.0`의 **cu126** wheel을 설치합니다.
GTX 1050은 compute capability 6.1입니다. 실제 확인된 wheel의 `compiled_architectures`에
`sm_61`이 있고 CUDA 행렬 연산·역전파가 성공해야 합니다.
`nvidia-smi`의 CUDA Version은 드라이버 지원 상한이며 torch wheel의 CUDA 버전과 동일할 필요는 없습니다.
이미 torch가 있으면 제거/재설치하지 않고 CUDA 진단부터 합니다. 호환되지 않으면 실패합니다.
Windows 서비스 계정의 PATH, 모델 캐시 및 네트워크 권한은 대화형 사용자 계정과 다를 수 있습니다.

CPU 전용 환경은 별도 폴더로 만들 수 있습니다.

```powershell
python -m venv .venv-transformer
& .venv-transformer/Scripts/python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
& .venv-transformer/Scripts/python.exe -m pip install -r transformer/configs/requirements-runtime.txt
& .venv-transformer/Scripts/python.exe -m pip install -r transformer/configs/requirements-dataset.txt
```

CUDA 환경을 이미 구성했다면 위 CPU 재설치 명령은 필요 없습니다. `--device cpu`만 사용하세요.

## PostgreSQL → Dataset

```powershell
& .venv-transformer/Scripts/python.exe -m transformer.dataset.build_dataset --output-dir data/training
Get-Content data/training/train.jsonl -TotalCount 1
```

기존 `pipeline.common.database.connect(read_only=True)`를 사용합니다.
저장소 루트 `.env` 또는 프로세스 환경변수의 `POSTGRES_HOST`, `POSTGRES_PORT`,
`POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, 선택적 `POSTGRES_SSLMODE`를 읽습니다.
자격증명은 출력하거나 데이터셋에 저장하지 않습니다.
명시한 `.env`를 사용하려면 `--env-file <경로>`를 지정하세요.

2026-09-17 실제 DB에서 확인한 `public.papers` 필드:
`paper_id bigint`, `title varchar(500)`, `keywords text[]`, `abstract/introduction/body/conclusion text`,
`language varchar(10)`, `doi varchar(255)`, `content_hash varchar(64)`.
저장소의 SQL 파일은 자리표시자여서 실제 서버의 컬럼을 조회해 확인했습니다.
Builder는 필요한 컬럼의 존재를 확인하고 SELECT만 실행합니다.
`training_samples`나 다른 테이블에 INSERT/UPDATE/DELETE/DDL을 실행하지 않습니다.

API Key 없이 동작하는 초기 데이터 생성 방식:

1. 영어 논문 중 세 절에 완결된 8~22단어 문장이 있는 논문을 선별합니다.
2. DOI·content hash·정규화 제목·본문·paper_id가 같은 항목을 전이적으로 묶습니다.
3. **논문 그룹을 먼저** seed 42로 80/10/10에 분할합니다. 각 분할 최소 1편이 필요합니다.
4. 그룹마다 대표 논문 하나를 선택하고 모든 원래 ID를 `source_paper_ids`로 기록합니다.
5. 절마다 주제 키워드와 겹치는 짧은 완결 문장 하나를 선택해 paper evidence를 만듭니다.
6. 선택한 세 문장을 Introduction/Body/Conclusion 형태로 정리해 짧은 target을 만듭니다.
7. topic은 상위 키워드, research question은 제목 기반 고정 템플릿으로 구성합니다.
8. 확실한 뉴스-논문 연결 정보가 없으므로 news evidence는 `[NO_NEWS_EVIDENCE]`입니다.

**정답은 원문 전체가 아닌 추출형 요약으로 만든 약한 학습 라벨입니다.**
주된 초기 학습 신호는 근거 활용과 초안 형식이며, 사람이 쓴 고품질 재서술 정답과 동등하지 않습니다.
입력의 근거에 정답 문장이 포함되는 것은 추출형 요약의 특성입니다.
그 때문에 loss가 낮아도 새로운 주장이나 다중 근거를 종합하는 능력을 입증하지 못합니다.
표현만 다른 중복 논문, 문맥 의존 문장, 잘못 추출된 절은 수동 검수가 필요합니다.
이후 검수된 입력/정답 쌍으로 교체해도 Trainer 인터페이스는 같습니다.

출력은 `train.jsonl`, `validation.jsonl`, `test.jsonl`, `manifest.json`입니다.
Manifest에 데이터 수, 제외 이유, seed, 컬럼, 각 파일 SHA-256을 기록합니다.
기존 스냅샷을 덮어쓰지 않습니다. 다시 export할 때는 새 `--output-dir`을 쓰세요.
다른 위치의 출력도 민감한 원문이므로 Git에 추가하지 마세요.
오프라인 DB export JSON 배열은 `--papers-json <파일>`로 읽습니다.
그 파일은 위 컬럼과 **실제 paper_id**를 포함해야 합니다. 수집기의 OpenAlex `id`와 혼용하지 마세요.

## Dry-run과 Full 학습

```powershell
& .venv-transformer/Scripts/python.exe -m transformer.training.train --dry-run --device cuda

# 위 명령 성공 후 동일 데이터·모델·설정으로 실행
& .venv-transformer/Scripts/python.exe -m transformer.training.train --device cuda --epochs 3
```

CPU는 두 명령 모두 `--device cpu`로 변경합니다. 로컬 기본 `auto`는 CUDA가 없으면 CPU를 선택하고
실제 device를 출력합니다. GPU workflow는 반드시 `--device cuda`여서 CPU로 조용히 넘어가지 않습니다.

| 설정 | 기본값 |
|---|---|
| 모델 | `google/flan-t5-small` (실제 resolved commit을 metadata에 저장) |
| 미세조정 | LoRA, q/v projection, rank 4, alpha 8, dropout 0.05 |
| batch / accumulation | 1 / 8 |
| input / target | 최대 384 / 128 tokens |
| epochs / learning rate | 3 / 0.0003 |
| 정밀도 | float32, fp16 꺼짐; Pascal에서 fp16 옵션 거부 |
| 메모리 | gradient checkpointing, cache off, Windows DataLoader workers 0 |
| CPU threads / seed | 8 / 42 |

모든 값은 `--model-name`, `--model-revision`, `--epochs`, `--batch-size`, `--learning-rate`,
`--max-input-length`, `--max-target-length`, `--gradient-accumulation-steps`, `--seed`,
`--lora-rank`, `--lora-alpha`, `--cpu-threads`, `--fp16`으로 설정합니다.
q/v projection을 쓰는 T5 계열 encoder-decoder 모델이 기본 대상입니다.
다른 아키텍처는 LoRA target module 호환성부터 확인하세요.

전체 원문을 모델에 넣지 않습니다. 제목·질문·지시·각 근거에 토큰 예산을 배분합니다.
길이를 줄여도 마지막 뉴스 필드를 일괄 삭제하지 않습니다.
너무 긴 target은 세 절 모두를 유지하는 방식으로 줄어들므로, 길이 설정을 바꾸면
tokenizer 결과를 반드시 검수하세요. 특히 48~96토큰 target은 문장을 자를 수 있습니다.

Dry-run은 train 최대 `max(8, 2*batch*accumulation)`건, validation 최대 2건, optimizer 2단계입니다.
검증 loss 전후 비교, 체크포인트 2개 저장, CPU에서 LoRA 병합 후 전체 모델 저장,
저장 모델의 재로딩과 예제 추론까지 성공해야 `dry_run_success.json`이 만들어집니다.
기본 위치는 `artifacts/transformer_dry_run/`입니다.

Full은 데이터 해시·모델 commit·설정(epochs 제외)·device·의존성·코드가 같은 dry-run receipt를 요구합니다.
다른 경로에서 시험했다면 `--dry-run-receipt <경로>/dry_run_success.json`을 지정하세요.
기존 출력이 있으면 새 `--output-dir`을 쓰세요. 모델을 자동으로 덮어쓰지 않습니다.
기본 full 모델 저장 위치는 `artifacts/transformer_model/`입니다.
test 데이터는 무결성·누수 검사만 하고 학습, 검증, 예제 생성에는 사용하지 않습니다.

산출물:

- 루트의 `model.safetensors`, `config.json`, tokenizer 파일: 독립적으로 로드 가능한 병합 모델
- `adapter/`: 별도 LoRA 가중치
- `checkpoints/checkpoint-*`: optimizer/scheduler/RNG/trainer 상태와 LoRA 체크포인트
- `training_metadata.json`: 설정, 버전, 모델 commit, 데이터 해시, 실제 device, loss, GPU peak allocation
- `training.log`, `sample_draft.txt`, `generation_diagnostics.json`
- 성공한 dry-run의 `dry_run_success.json`, 실패 시 가능한 경우 `failure.json`

Dry-run의 결과물은 운영 모델이 아닙니다. loss 감소, 형식 보정, GPU 실행 성공은 생성 품질 합격과 다릅니다.

## 추론 CLI

Evidence 파일은 문자열 배열 JSON입니다. `paper-evidence.json` 예:

```json
["Participants shared translations and coordinated community activities online."]
```

```powershell
& .venv-transformer/Scripts/python.exe -m transformer.inference.generate `
  --model-path artifacts/transformer_model `
  --title "Digital participation in music fandoms" `
  --topic "Online fandom participation" `
  --research-question "How does online participation support community activity?" `
  --paper-evidence-file paper-evidence.json `
  --device cpu
```

선택적으로 `--news-evidence-file news-evidence.json`, `--instruction`, `--output <파일>`을 전달합니다.

## GitHub Actions에서 누를 순서

1. 이 변경 파일들을 commit/push해 `train-transformer.yml`이 기본 브랜치에 보이게 합니다.
2. **Settings → Actions → Runners**에서 Windows runner가 Online이며
   `self-hosted`, `Windows`, `X64`, `gpu-train` 라벨을 모두 갖는지 확인합니다.
3. DB에서 새 export를 할 경우 기존 `ec2-db` runner도 Online이어야 합니다.
   기존 POSTGRES_PORT/DB/USER/PASSWORD repository secrets를 재사용합니다.
   PostgreSQL의 외부 포트를 새로 열 필요는 없습니다.
4. **Actions → Train transformer → Run workflow**를 선택합니다.
5. 처음에는 `mode=dry-run`, `epochs=3`, dataset 입력 두 개는 빈 값으로 실행합니다.
6. EC2 job이 DB를 읽어 JSONL artifact를 만들고 Windows job이 내려받습니다.
   Windows job은 Python 확인 → nvidia-smi → 재사용 venv → CUDA 커널 확인 → dry-run 순서입니다.
7. Windows job 성공 후 Artifacts에서 `transformer-dry-run-<run_id>-<attempt>`를 받아
   metadata, loss, 생성 결과 및 diagnostics를 확인합니다.
8. Full 실행은 `mode=full`로 새로 실행합니다. 직전과 같은 데이터를 쓰려면 직전의
   `dataset_run_id`와 정확한 `transformer-dataset-<run_id>-<attempt>` artifact 이름을 입력합니다.
   이 경우 DB export job을 건너뛰고 dataset artifact만 받습니다.
9. Full 실행도 시작 시 같은 설정으로 GPU dry-run을 다시 수행한 후 full을 실행합니다.
   성공한 `transformer-full-...` artifact의 `model/`이 최종 병합 모델입니다.

이번 변경만 올리는 명령 (저장소 루트):

```powershell
git add -- .gitignore .github/workflows/train-transformer.yml transformer tests/transformer
git diff --cached --check
git commit -m "Add lightweight transformer first-draft training"
git pull --no-rebase origin main
git push origin main
```

각 명령이 성공한 뒤 다음 명령을 실행하세요. 원격을 반영할 때 새 충돌이 발생하면
push로 진행하지 말고 충돌을 확인하세요. 강제 push는 사용하지 않습니다.

Windows steps는 `powershell`을 사용합니다. 가상환경은 checkout 청소 대상 밖인
`RUNNER_TOOL_CACHE/paper-transformer-py312-cu126`에 재사용합니다.
학습 job에는 PostgreSQL 비밀번호를 전달하지 않습니다.
일정 실행이나 PR trigger는 없으며 수동 실행만 지원합니다.
데이터와 모델 artifacts는 14일 보관합니다. 이전 artifact가 만료되면 새 DB export로 실행하세요.

## 문제 확인 및 테스트

- Runner 대기: label/Online 상태 및 `ec2-db` export job을 확인합니다.
- Python/CUDA 오류: setup 출력, torch version, torch CUDA, compiled_architectures, GPU 이름을 봅니다.
- OOM: GPU 사용 앱 종료, batch 1 유지, 길이를 256/96으로 줄이고 새 dry-run을 수행합니다.
- DB 오류: 누락 환경변수 이름과 안전한 오류 분류만 출력됩니다. 자격증명을 로그에 붙이지 마세요.
- loss NaN/Inf: fp16을 끄고 데이터와 설정을 확인합니다. 이 경우 성공 receipt가 생성되지 않습니다.
- 형식 누락/거부: `sample_draft.txt`와 `generation_diagnostics.json`을 확인합니다.
- 데이터 누수/변경: paper_id, source_paper_ids, manifest SHA-256 오류를 해결하고 새 스냅샷을 만듭니다.

```powershell
& .venv-transformer/Scripts/python.exe -m pip install pytest pyyaml
& .venv-transformer/Scripts/python.exe -m pytest tests/transformer -q
```

공식 참고: [PyTorch cu126 2.8 설치](https://pytorch.org/get-started/previous-versions/),
[FLAN-T5-small](https://huggingface.co/google/flan-t5-small),
[Transformers 4.56.2 Trainer](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/trainer),
[PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora).
