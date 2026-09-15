# 논문 생성 AI Agent — Day 1 작업 정리

작성일: 2026-09-15

---

## 1. 매일 체크리스트

| 항목                                  | 상태            | 담당  |
| ----------------------------------- | ------------- | --- |
| 논문/뉴스 수집·전처리 Pipeline 구현            | ✅             | 박수암 |
| GitHub Actions 자동 수집 Workflow 구현    | ✅             | 박수암 |
| 뉴스 약 300건 실제 수집 확인                  | ✅             | 박수암 |
| PostgreSQL 실제 적재                    | 🔴 오류 수정 중    | 박수암 |
| OpenAlex 논문 실제 수집                   | 🔴 0건 문제 확인 중 | 박수암 |
| 1536차원 Embedding 기능 구현              | ✅             | 이혜림 |
| Paper / News Vector RAG 구현          | ✅             | 이혜림 |
| Neo4j Graph RAG 구현                  | ✅             | 이혜림 |
| Hybrid RAG 및 Evidence Validation 구현 | ✅             | 이혜림 |
| Agent 전체 Pipeline 구현                | ✅             | 이혜림 |
| FastAPI Backend 기본 API 구현           | ✅             | 이혜림 |
| PostgreSQL / Neo4j 실제 연결 확인         | ✅             | 이혜림 |
| Agent / Backend Mock Test           | ✅             | 이혜림 |
| 실제 데이터 기반 RAG E2E                   | ⬜ 데이터 적재 후 진행 | 공동  |
| 실제 Transformer 연결                   | ⬜ Day 2 예정    | 공동  |
| GPU 환경 구성                           | ⬜ Day 2 예정    | 이혜림 |

---

## 2. 박수암 — Data / Pipeline / Transformer 준비

### 완성

* 논문/뉴스 수집·전처리·중복 제거·DB Loader 코드 정리
* 실제 Git 프로젝트 구조에 Pipeline 코드 통합
* 뉴스 수집 Pipeline 구현

  * 뉴스 수집
  * 본문 전처리
  * 중복 제거
  * 한국어 원문 보존
  * 영어 RAG용 번역본 생성
  * Category 자동 분류
  * Keywords 자동 분류
* 뉴스 약 300건 실제 수집 및 본문 전처리 확인
* 논문 OpenAlex 기반 수집 구조 구현
* `papers` / `news` PostgreSQL Loader 구현
* 뉴스 Pipeline 단위 테스트 통과
* 논문 Pipeline 단위 테스트 통과

### GitHub Actions

* `ingest-news.yml` 구현

  * 뉴스 수집
  * 전처리
  * 중복 제거
  * 영문 번역
  * PostgreSQL 적재 자동화
* `ingest-papers.yml` 구현

  * OpenAlex 논문 수집
  * 전처리
  * PostgreSQL 적재 자동화
* GitHub Repository Secrets 등록
* 작업 브랜치 `suam/ingestion-v2`
* Commit `5873cefeeb2ae4b4bab89fe81b286ca1aa62a392`
* Push 완료
* `main` 병합 완료

### 테스트 / 검증

* 뉴스 Pipeline 단위 테스트 ✅
* 논문 Pipeline 단위 테스트 ✅
* OpenAI 뉴스 번역 단계 확인 ✅
* OpenAlex API 호출 단계 확인 ✅
* 뉴스 Category / Keywords 자동 분류 확인 ✅
* 뉴스 약 300건 수집 확인 ✅
* GitHub Actions 전체 통합 실행 🟡
* PostgreSQL 실제 적재 🔴
* OpenAlex 논문 실제 수집 🔴

### 진행 중

* GitHub Actions 뉴스 → PostgreSQL 적재 오류 수정
* `DatabaseConfigurationError` 원인 확인
* OpenAlex 논문 수집 결과 0건 문제 수정
* 수집 뉴스 데이터 품질 검증
* 뉴스/논문 Actions 전체 Pipeline 재실행

---

## 3. 이혜림 — RAG / Agent / Backend

### Embedding / RAG 데이터 처리

* OpenAI `text-embedding-3-small` 기반 1536차원 Embedding 구현
* 실제 Embedding 호출 및 1536차원 반환 확인
* 논문 / 뉴스 Chunk 생성 기능 구현
* Chunk → Embedding → `paper_chunks` / `news_chunks` 적재 구조 구현
* 논문과 뉴스 Vector RAG 구조 분리

  * `paper_rag`
  * `news_rag`
* pgvector Cosine Similarity 기반 Top-K 검색 구현
* Graph에서 새로 발견된 논문의 실제 Evidence Chunk 재조회 기능 구현

### Graph RAG

* Neo4j 연결 코드 구현
* 실제 Neo4j Bolt 연결 테스트 성공
* PostgreSQL `paper_id` ↔ Neo4j `Paper.paper_id` 연동 구조 구현
* 관계 기반 관련 논문 검색 구현
* PostgreSQL → Neo4j Sync 구조 구현
* Paper / Author / Keyword / Source Graph 생성 구조 구현
* `graph_synced` 기반 동기화 상태 관리 구조 구현
* 기존 프로젝트 구조에 맞게 Graph 코드를

  * `builder`
  * `sync`
  * `graph_rag`
    영역으로 정리

### Hybrid RAG

* Vector RAG + Graph RAG 통합
* RRF 기반 Hybrid Ranking 구현
* Vector / Graph 중복 논문 통합
* Graph-only 논문 Evidence 보충
* Hybrid RAG Mock 테스트 완료

### Evidence Validation / Abstention

* 실제 Evidence Chunk 존재 여부 검사
* 최소 Evidence 논문 수 검사
* Similarity 기준 검사
* 최소 Evidence Chunk 수 검사
* 근거 충분 → `generate`
* 근거 부족 → `abstain`
* Generate / Abstain Mock 테스트 완료

### Agent Pipeline

* 한국어 논문 제목 / 주제 → 영문 Query 변환
* 영문 Query → Hybrid RAG 연결
* Hybrid RAG → Evidence Validation 연결
* 근거 부족 시 Transformer / LLM 호출 차단
* Transformer Adapter 구현
* 실제 Transformer 미완성 상태 대응 Mock 구조 구현
* RAG Evidence → Transformer 전달 형식 구현
* Transformer 초안 + RAG Evidence 기반 LLM Finalizer 구현
* 영문 Introduction / Body / Conclusion 생성 구조 구현
* 최종 영문 → 한국어 변환
* 사용자 입력 논문 제목 유지
* 최종 출력:

  * 제목
  * 서론
  * 본론
  * 결론
* 최종 한국어 결과 4,500자 제한 및 재압축 기능 구현
* 전체 Agent Pipeline 연결

### Backend

* FastAPI 기본 구조 연결
* `POST /api/generate` 구현
* Agent Pipeline ↔ Backend Service 연결
* Completed / Abstained 응답 처리
* Request / Response Schema 구현
* `GET /health` 구현
* `GET /ready` 구현
* PostgreSQL 연결 상태 확인 API 구현
* Neo4j 연결 상태 확인 API 구현
* 잘못된 입력 Validation 처리
* Swagger 기반 API 실행 구조 확인

### 테스트 / 검증

* PostgreSQL `paper_agent` 실제 연결 ✅
* pgvector 활성화 확인 ✅
* Neo4j 실제 연결 ✅
* Agent 정상 생성 분기 Mock Test ✅
* Agent Abstain 분기 Mock Test ✅
* Agent Mock Test `2 passed` ✅
* Backend `/health` Test ✅
* Backend `/ready` 정상 / 장애 Test ✅
* `/api/generate` Completed Mock Test ✅
* `/api/generate` Abstained Mock Test ✅
* 빈 제목 Validation Test ✅
* Backend Mock Test `6 passed` ✅
* Agent + Backend Mock Test 전체 통과 ✅
* Circular Import 문제 해결 ✅
* 프로젝트 기존 폴더 구조에 맞게 코드 정리 ✅

---

## 4. 팀 전체 Pipeline 현재 상태

현재 전체 구조:

`논문 / 뉴스 수집`
→ `정제 / 중복 제거 / 번역`
→ `PostgreSQL papers / news`
→ `Chunk 생성`
→ `1536차원 Embedding`
→ `paper_chunks / news_chunks`
→ `Vector RAG`
→ `PostgreSQL → Neo4j Sync`
→ `Graph RAG`
→ `Hybrid RAG`
→ `Evidence Validation`
→ `Transformer`
→ `LLM Finalizer`
→ `한국어 변환`
→ `4,500자 제한`
→ `FastAPI /api/generate`

### 현재 구현 상태

| 영역                  | 상태            |
|---------------------|---------------|
| 뉴스 수집               | ✅ 실제 수집 확인    |
| 뉴스 전처리              | ✅             |
| 뉴스 영문 번역            | ✅             |
| 뉴스 분류 / 키워드         | ✅             |
| 뉴스 중복 제거            | ✅             |
| 논문 수집 구조            | ✅             |
| PostgreSQL Loader   | ✅ 코드 구현       |
| GitHub Actions      | ✅ Workflow 구현 |
| PostgreSQL 실제 적재    | 🔴 오류 수정 중    |
| OpenAlex 실제 논문 수집   | 🔴 수정 중       |
| Embedding           | ✅ 구현 완료       |
| Chunking            | ✅ 구현 완료       |
| Vector RAG          | ✅ 구현 완료       |
| Graph RAG           | ✅ 구현 완료       |
| Hybrid RAG          | ✅ 구현 완료       |
| Evidence Validation | ✅ 구현 완료       |
| Agent Pipeline      | ✅ 구현 완료       |
| Backend API         | ✅ 구현 완료       |
| Mock E2E Test       | ✅             |
| 실제 데이터 RAG Test     | ⬜ 데이터 대기      |
| 실제 Transformer 연결   | ⬜ Day 2       |
| GPU Inference       | ⬜ Day 2       |
| 실제 전체 E2E           | ⬜ Day 2       |
| Frontend 구현         | ⬜ Day 3       |
| 최종 테스트              | ⬜ Day 4       |

---

## 5. 팀 연동 지점

### Data → RAG

박수암 담당:

`수집`
→ `전처리`
→ `papers / news 적재`

이혜림 담당:

`papers / news`
→ `Chunk`
→ `Embedding`
→ `paper_chunks / news_chunks`
→ `Vector RAG`

### PostgreSQL → Graph RAG

`PostgreSQL papers.paper_id`
→ `Neo4j Paper.paper_id`
→ `Author / Keyword / Source 관계 생성`
→ `Graph RAG`

### RAG → Transformer

`Hybrid RAG Evidence`
→ `generate_draft(title, topic, evidence)`
→ Transformer 영문 초안

### Transformer → Agent

`Transformer 초안 + RAG Evidence`
→ LLM Finalizer
→ Introduction / Body / Conclusion
→ 한국어 변환
→ 4,500자 이하 최종 초안

### Agent → Backend

`POST /api/generate`
→ Agent Pipeline
→ 최종 논문 반환

---

## 6. 오늘 발견된 막힌 이슈

### 1. PostgreSQL 적재 오류

* **담당:** 박수암
* **현상:** 뉴스 Pipeline PostgreSQL Loader 단계에서 `DatabaseConfigurationError`
* **확인 대상:**

  * `pipeline/common/database.py`
  * GitHub Repository Secrets
  * PostgreSQL 접속 설정
* **필요 조치:** DB 연결 설정 수정 후 Actions 재실행

### 2. OpenAlex 논문 수집 0건

* **담당:** 박수암
* **현상:** 논문 다운로드 대상 0건
* **확인 대상:**

  * OpenAlex API Key
  * API 요청 방식
  * Query / Filter 조건
* **필요 조치:** 요청 방식 수정 후 실제 논문 수집 재실행

### 3. 실제 RAG 테스트 대기

* **담당:** 이혜림
* **현상:** PostgreSQL `papers / news` 데이터가 아직 없어 실제 Vector / Hybrid 검색 검증 불가
* **필요 조치:** Data Pipeline 적재 완료 후 즉시 Chunk / Embedding 및 실검색 수행

### 4. 실제 Transformer 연결 대기

* **담당:** 공동
* **현상:** 현재 Agent는 Transformer Adapter + Mock 상태
* **필요 조치:** 실제 `generate_draft(title, topic, evidence)` Inference 연결

---

## 7. Git / 반영 상태

### 박수암

* 브랜치: `suam/ingestion-v2`
* Commit: `5873cefeeb2ae4b4bab89fe81b286ca1aa62a392`
* Push: ✅
* Main 반영: ✅

주요 반영:

* 논문 / 뉴스 수집 Pipeline
* 뉴스 전처리 / 중복 제거
* 뉴스 영문 번역
* PostgreSQL Loader
* GitHub Actions
* Pipeline Test

### 이혜림

* 브랜치: `hyerim`
* Push: ✅
* Main 반영:

  * RAG Core ✅
  * Agent Pipeline ✅
  * Backend ✅

주요 반영:

* Embedding
* Vector / Graph / Hybrid RAG
* Evidence Validation
* Agent Pipeline
* Mock E2E Test
* Backend API
* DB 상태 확인 API

### 보안

* `.env` Git 제외 확인
* OpenAI API Key Git 제외
* PostgreSQL Password Git 제외
* Neo4j Password Git 제외

---

## 8. Day 2 진행 전 확인 사항

### 박수암

1. PostgreSQL `DatabaseConfigurationError` 해결
2. 뉴스 GitHub Actions → PostgreSQL 실제 적재 성공
3. OpenAlex API 요청 오류 해결
4. 논문 실제 수집 / 원문 확보
5. 뉴스 / 논문 Actions 전체 Pipeline 재실행
6. DB 적재 완료 후 이혜림에게 공유
7. `training_samples` 생성 준비
8. Transformer Dataset / Fine-Tuning 진행
9. `generate_draft(title, topic, evidence)` Inference 제공

### 이혜림

1. GPU 설치 및 인식 확인
2. `nvidia-smi`
3. PyTorch / CUDA 환경 설정
4. `torch.cuda.is_available()` 확인
5. `papers / news` 실제 데이터 확인
6. 논문 / 뉴스 Chunk 생성
7. 1536차원 Embedding 실제 생성
8. `paper_chunks / news_chunks` 적재
9. PostgreSQL → Neo4j 실제 Sync
10. Neo4j Browser Graph 확인
11. Paper Vector RAG 실제 검색
12. News Vector RAG 실제 검색
13. Graph RAG 실제 검색
14. Hybrid RAG 실제 검색
15. Similarity / Evidence Validation 기준 조정
16. Mock Transformer → 실제 Transformer 교체
17. 실제 Transformer GPU Inference 확인
18. `/api/generate` 실제 E2E 테스트

---

## 9. Day 2 공동 완료 목표

Day 2 종료 시 아래 전체 흐름을 실제 데이터와 실제 모델로 성공시키는 것을 목표로 함.

`실제 수집 데이터`
→ `PostgreSQL`
→ `Chunk`
→ `Embedding`
→ `Vector RAG`
→ `Neo4j`
→ `Graph RAG`
→ `Hybrid RAG`
→ `Evidence Validation`
→ `Transformer`
→ `LLM`
→ `한국어 최종 초안`
→ `FastAPI /api/generate`

### 최종 검증 Case

* 정상 주제 → `completed`
* 근거 부족 주제 → `abstained`
* 최종 결과 → 제목 / 서론 / 본론 / 결론
* 최종 한국어 결과 → 4,500자 이하
* PostgreSQL / Neo4j → `/ready` 정상
* GPU Transformer Inference → 정상

---

## 10. Day 1 종료 시 팀 전체 상태

### 박수암

* 진행률: 약 70%
* 오늘 목표: 일부 완료
* 주요 블로커:

  * PostgreSQL 적재 오류
  * OpenAlex 논문 수집 0건

### 이혜림

* 오늘 목표: ✅ 완료
* 추가 선행 작업: ✅ Backend 기본 구현까지 완료
* Mock 기반 개발 및 검증 가능한 범위: ✅ 완료
* 남은 핵심:

  * 실제 데이터 통합
  * GPU
  * 실제 Transformer 연결

### 팀 전체

**Day 1 핵심 구조 구현 완료**

현재 수집 측 실제 데이터 적재 이슈 해결 후 RAG / Transformer / Backend 전체 통합 테스트가 가능한 상태.

**Day 2 핵심 목표:**
실제 데이터 + 실제 Transformer + GPU 기반 End-to-End Pipeline 완성.
