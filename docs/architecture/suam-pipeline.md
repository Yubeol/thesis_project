# 박수암 데이터 파이프라인

## 작업 범위와 협업

논문·뉴스 수집, 전처리, 기존 PostgreSQL 적재, 논문 텍스트 기반 학습 데이터셋,
Pretrained Transformer GPU Fine-Tuning, 모델·Tokenizer·inference 제공이 담당 범위다.

- 구현: `pipeline/papers/`, `pipeline/news/`, `pipeline/common/`, `transformer/`
- 검증: `tests/pipeline/`, `tests/transformer/`
- 워크플로: `ingest-papers.yml`, `ingest-news.yml`, `train-transformer.yml`
- 설명서: 이 문서와 향후 `docs/api/transformer-inference.md`
- 원본·가공 데이터: `data/`, 모델 산출물: `models/`, `transformer/checkpoints/`
- `rag/`, `agent/`, `backend/`, `frontend/`, `database/` 및 다른 워크플로는 수정하지 않는다.
- 공용 루트 설정을 수정하는 대신 담당 의존성은 `pipeline/common/requirements.txt`에 둔다.
- 기존 DB 스키마를 먼저 확인하고 적재 코드를 맞춘다. 테이블 생성·변경은 이 파이프라인의 동작이 아니다.
- 뉴스 및 embedding vector는 Transformer 학습 데이터에서 제외한다.

작업 브랜치는 `suam/paper-pipeline`이다. 작업 위치는 바깥쪽 `paper-ai-agent` 저장소다.
하위 `thesis_project/`는 별도 Git 복제본이므로 통째로 stage하거나 submodule로 등록하지 않는다.
우리 변경 파일을 명시하여 커밋하고 별도 브랜치의 PR로 통합한다.

## 논문 수집·전처리 실행

Python 3.12 환경에서 저장소 루트를 작업 디렉터리로 사용한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r pipeline/common/requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests/pipeline -p 'test_paper*.py' -v
.\.venv\Scripts\python.exe -m pipeline.papers.collector.openalex
.\.venv\Scripts\python.exe -m pipeline.papers.collector.download
.\.venv\Scripts\python.exe -m pipeline.papers.extractor.process
```

`OPENALEX_API_KEY`는 선택적인 환경변수다. 코드에 넣지 않는다.
GitHub에서는 동일한 이름의 Actions Secret으로 설정한다. 현재 수집 CLI는 `.env`를
자동 로드하지 않으므로 필요하면 실행 환경에서 이 변수를 제공한다.

### 수집

`pipeline/papers/collector/config.json`의 주제 6개를 OpenAlex에서 검색한다.
철회 논문을 제외하고 OA 논문을 우선 수집한다. 제목·초록에서 K-pop/Korean wave/Hallyu와
팬덤·SNS·글로벌 확산·팬 활동 맥락을 확인한다. DOI·정규화 제목·OpenAlex ID로 중복을 제거한다.
이는 자동 관련성 선별이며 개별 논문의 학술적 신뢰도 평가를 대체하지 않는다.

API 응답은 요청별로 보존하고 같은 설정의 재실행은 로컬 캐시를 사용한다.
최신 데이터를 다시 조회할 때 `--refresh`를 사용한다.
출처 URL, 저자, 연도, 원어, 초록, OA 상태, 위치별 라이선스와 검색 출처를 보존한다.
메타데이터만 있는 단계에서 서론·본론·결론은 null로 둔다.

다운로드는 OA 위치와 HTML의 `citation_pdf_url`을 사용한다. 제한된 재시도와
호스트별 접근 간격을 적용하며, 접근 제한은 실패 목록에 남긴다.
원문 응답과 SHA-256, 원래 URL, 최종 URL, 라이선스 출처를 보존한다.
PDF는 파일 시그니처를 확인한다. HTML에는 랜딩 페이지나 안내문이 포함될 수 있으므로
다운로드 성공을 본문 확보나 학습 가능 판정과 동일하게 취급하지 않는다.

### 전처리

PDF는 pypdf, HTML은 trafilatura로 추출한다. 변경 전 페이지별 텍스트도 별도 보존한다.
반복 머리말·꼬리말·페이지 번호를 정리하고 실제 제목으로 확인되는 구간을 분리한다.
한국어·영어·일부 인도네시아어 제목을 지원한다. 없는 구간은 만들어 채우지 않는다.
제목 구간의 offset은 정제된 `fulltext` 기준이다.

문서 길이, 제목 일치, 서론·본론·결론 최소 길이, 문자 손상, 본문 언어를 검사한다.
메타데이터 언어와 실제 본문 언어는 다를 수 있어 두 값을 모두 보존한다.
`training_eligible`은 자동 구조 검사 결과다. 최종 학습 승인이나 품질 보증을 뜻하지 않는다.
제목 탐지, 문단 순서, 표·각주 혼입, 언어, 출처와 사용 조건은 학습 코퍼스 확정 전에 검토해야 한다.

### 결과 파일

| 경로 | 내용 |
|---|---|
| `data/raw/papers/api/` | 원본 OpenAlex API 응답 |
| `data/raw/papers/papers.json` | 중복 제거한 논문 메타데이터 |
| `data/raw/papers/collection_report.json` | 검색·관련성·중복 제거·오류 통계 |
| `data/raw/papers/documents/<ID>/` | 원문 파일과 출처·해시 manifest |
| `data/raw/papers/download_report.json` | 원문 확보 성공·실패 내역 |
| `data/processed/papers/extractions/<ID>.json` | 페이지별 원본 텍스트와 구조 추출 |
| `data/processed/papers/papers.json` | DB 적재 전 논문 레코드 |
| `data/processed/papers/processing_report.json` | 구조 검사 통계·추출 실패 내역 |

데이터 파일은 기존 `.gitignore`에 의해 일반 Git 커밋에서 제외된다.
GitHub Actions는 매주 월요일 00:00 UTC(09:00 KST)와 수동 실행을 지원한다.
실패 시에도 생성된 로그·데이터를 artifact로 보존하며 보관 기간은 30일이다.
장기 코퍼스 보관과 이후 데이터셋 단계로의 전달은 추가로 연결해야 한다.

## 검증된 진행 상태 — 2026-09-15

현재 로컬 실행으로 확인한 결과이며 GitHub 서버에서 워크플로를 실행한 결과는 아니다.

| 요구 사항 | 현재 증거 / 남은 작업 |
|---|---|
| 논문 약 50~70편 | 600개 검색 결과에서 고유 후보 134개, 메타데이터 70건 선정 |
| 실제 논문 원문 | PDF 38건, HTML 9건, 다운로드 실패 23건. 추가 확보 필요 |
| 논문 구조 전처리 | 텍스트 44건 추출, 자동 구조 검사 통과 23건(영어 21, 인도네시아어 2). 코퍼스 전체 수동 검토 미완료 |
| 단위 테스트 | 수집·중복 제거·실패 처리·캐시·구조 추출 등 16개 통과 |
| 뉴스 약 200~300건 | 미구현. GDELT 접근 시험에서 rate limit 및 결과 미확보. 공급원 검증 필요 |
| PostgreSQL papers/news 적재 | 미구현·미실행. 사용자 지정 내부 복제본의 `.env`가 아직 없고 SQL은 자리표시자 상태 |
| training_samples / Train·Validation·Test | 미구현. 논문 ID 단위 분리로 동일 논문 구간 간 데이터 누수 방지 필요 |
| Pretrained Transformer 선정 | 미완료. Tokenizer 길이와 실제 코퍼스 언어를 확인하여 선택 |
| GPU Fine-Tuning | 미실행. 로컬 GPU GTX 1050 2GB 확인, 실제 모델의 학습 가능성 검증 필요 |
| Model / Tokenizer / Config | 학습 산출물 없음 |
| generate_draft(title, topic, evidence) | 미구현. 팀원이 제공하는 evidence를 입력받고 검색기는 구현하지 않음 |

다음 작업은 뉴스 공급원·수집 구현과 논문 원문 확보 보완이다. 실제 DB 설정·스키마가
제공되면 변경 없이 연결을 검증하고 적재를 구현한다. 데이터셋은 논문 텍스트만 사용하며,
모델과 Tokenizer를 선정한 뒤 입력·출력 길이에 맞춰 구성한다.

## 참고한 공식 자료

- [OpenAlex API](https://help.openalex.org/api/)
- [OpenAlex 인증](https://help.openalex.org/api/authentication/)
- [Crossref metadata와 fulltext 구분](https://www.crossref.org/documentation/retrieve-metadata/text-and-data-mining/)
- [pypdf 텍스트 추출 및 한계](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)
- [trafilatura Python API](https://trafilatura.readthedocs.io/en/latest/usage-python.html)
- [GitHub Actions artifacts](https://docs.github.com/en/actions/tutorials/store-and-share-data)
