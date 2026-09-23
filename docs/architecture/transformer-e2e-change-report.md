# Transformer 문장 품질 및 Orchestrator E2E 연동 점검 보고서

> 이 문서는 당시 점검 결과를 기록한 이력 문서다. 이후 보강 사항은 현재 코드와 테스트를 기준으로 확인한다.

작성 목적: Transformer 초안의 문장 깨짐 문제를 재현하고, 전체 논문 생성 파이프라인에서 안전하게 처리하기 위해 수행한 변경과 팀원 코드에 미치는 영향을 공유한다.

## 1. 작업 요약

- 원격 `main`의 최신 Orchestrator, LLM2, LLM3 변경사항을 로컬에 반영했다.
- 제공받은 K-pop/TikTok 예제로 실제 Orchestrator E2E를 실행했다.
- 기존 Transformer 모델과 현재 추론 코드의 학습 계약이 서로 다르다는 점을 확인했다.
- 불량하거나 계약이 맞지 않는 Transformer 출력을 그대로 Finalizer에 전달하지 않도록 근거 기반 대체 초안을 추가했다.
- Finalizer가 RAG 근거에 없는 숫자를 생성하는 경우 자동으로 한 번 재작성하고, 반복되면 결과를 차단하도록 검증을 추가했다.
- 전체 테스트 97개와 실제 E2E 실행을 통과했다.

## 2. 확인된 원인

기존 로컬 학습 모델은 예전의 “전체 초안을 한 번에 생성”하는 입력·출력 형식으로 학습됐다. 이후 추론 코드는 `Introduction`, `Body`, `Conclusion`을 각각 생성하는 섹션 단위 방식으로 변경됐다.

따라서 현재 모델 파일과 최신 추론 코드의 프롬프트 및 출력 계약이 일치하지 않는다. 실제 실행에서는 다음 문제가 발생했다.

- 세 섹션 모두 근거 검증에서 탈락해 근거 부족 문구로 대체되는 경우
- 제목만 세 번 반복하는 경우
- 문장이 지나치게 짧거나 동일한 섹션이 반복되는 경우
- 검색 결과를 전부 전달해 384-token 입력에서 각 Evidence에 몇 토큰밖에 배정되지 않는 경우
- PDF 원문에 포함된 붙어 있는 단어, 잘린 문장, 인용 표식이 초안에 노출되는 경우

최신 섹션 단위 데이터 169편, 학습 405개, 검증 51개, 테스트 51개 샘플로 후보 모델을 재학습했으나 검증 손실은 약 3.58이었고 일부 섹션이 여전히 근거 검증에서 탈락했다. 따라서 후보 모델은 운영 모델로 승격하지 않았다.

## 3. 박수암 담당 영역에서 수행한 변경

| 파일 | 변경 내용 | 영향 |
|---|---|---|
| `agent/draft_generator/generator.py` | 모델 메타데이터의 프롬프트 코드 해시와 현재 코드를 비교해 학습 계약 일치 여부를 검사 | 오래된 모델을 최신 코드로 잘못 실행하는 상황 방지 |
| `agent/draft_generator/grounded_fallback.py` | 근거 문장 기반 대체 초안 생성, 반복 제목·짧은 섹션·근거 부족·깨진 문장 감지 | Transformer가 실패해도 근거 밖 사실을 새로 만들지 않는 초안 제공 |
| `agent/grounding.py` | 최종문 숫자와 Paper/News Evidence의 숫자를 대조 | 근거에 없는 수치 검출 |
| `scripts/test/test-orchestrator.ps1` | UTF-8 환경을 설정하고 전체 Orchestrator를 실행하는 재현 스크립트 추가 | Windows 터미널 한글 깨짐 방지 및 E2E 재현 간소화 |
| `tests/transformer/*` | 현재 섹션 단위 Dataset/Tokenization 규격에 맞게 테스트 갱신 | 오래된 전체 초안 규격을 검사하던 테스트 수정 |
| `tests/agent/test_grounding.py` | 대체 초안과 근거 없는 숫자 재작성 테스트 추가 | 품질 방어 로직 회귀 방지 |
| `pytest.ini` | 테스트 탐색 범위를 `tests` 폴더로 제한 | `artifacts` 안의 백업 저장소 테스트가 중복 수집되는 문제 방지 |

공개 함수 `generate_transformer_draft()`의 인자와 문자열 반환 규격은 변경하지 않았다.

## 4. 팀원 영역에 영향을 주는 연동 변경

다음 변경은 Transformer를 전체 파이프라인에서 검증하기 위해 필요했지만, 팀원 담당 파일에 해당하므로 병합 전 확인이 필요하다.

| 파일 | 변경 내용 | 호환성 및 주의사항 |
|---|---|---|
| `agent/orchestrator.py` | 전체 Evidence는 LLM2/LLM3에 유지하고, Transformer에는 유사도 상위 Paper 4개와 News 2개만 전달 | 반환 JSON 구조와 공개 함수 인자는 변경하지 않음 |
| `agent/evidence.py` | Evidence를 similarity 내림차순으로 정렬하고 선택 개수 제한 옵션 추가 | `max_papers`, `max_news`는 선택 인자이므로 기존 호출은 가능하지만 Evidence 순서는 유사도 기준으로 바뀔 수 있음 |
| `agent/finalizer/finalizer.py` | 최종문에 근거 없는 숫자가 있으면 한 번 재작성하고, 다시 발생하면 `RuntimeError`로 차단 | 문제가 있을 때 OpenAI API 호출이 한 번 추가될 수 있음 |
| `rag/graph/graph_rag/retriever.py` | 존재하지 않는 Neo4j 관계 타입을 Cypher에 직접 선언해 경고가 발생하던 방식을 파라미터 필터 방식으로 변경 | 검색 대상 관계 의미는 동일하며 경고만 제거 |
| `tests/agent/test_agent_pipeline_mock.py` | 과거 Orchestrator API를 모킹하던 테스트를 현재 `generate_paper()` 흐름으로 갱신 | 운영 코드 변경 없음 |

FastAPI 코드, 데이터베이스 스키마, Migration, 테이블 데이터, `.env`는 수정하지 않았다.

## 5. 실제 검증 결과

### 자동 테스트

```text
97 passed, 3 warnings
```

경고 3개는 Starlette와 tokenizer 라이브러리의 DeprecationWarning이며 테스트 실패는 아니다.

### 실제 Orchestrator E2E

사용 주제:

```text
K-pop의 글로벌 확산에서 TikTok이 미친 영향
```

최종 확인 결과:

- `allowed=True`
- 1차 Paper/News RAG 성공
- Adaptive RAG 실행 성공
- Transformer 계약 불일치 감지 성공
- 오래된 모델 실행을 건너뛰고 근거 기반 초안 생성
- LLM2 Gap Analysis 성공
- LLM3 Finalizer 성공
- 최종 한국어 `서론 / 본론 / 결론` 출력 성공
- Neo4j의 존재하지 않는 관계 타입 경고 제거 확인
- 근거에 없는 숫자가 남은 경우 결과가 통과하지 않도록 검증

재현 명령:

```powershell
.\scripts\test\test-orchestrator.ps1
```

다른 주제는 다음과 같이 실행할 수 있다.

```powershell
.\scripts\test\test-orchestrator.ps1 `
  -Topic "확인할 주제" `
  -Instruction "학술적인 문체로 작성"
```

## 6. 현재 한계

- 서비스의 E2E 흐름과 불량 출력 방어는 동작하지만, Transformer 모델 자체의 생성 품질이 완성된 것은 아니다.
- 현재 기본 모델 파일은 최신 섹션 단위 코드와 계약이 다르므로 직접 실행하지 않는다.
- 후보 모델을 최신 규격으로 다시 학습했지만 품질 기준을 충족하지 못해 운영 경로로 승격하지 않았다.
- 현재 대체 초안은 RAG 원문 문장을 보수적으로 추출하므로 원문 품질이 낮으면 표현이 매끄럽지 않을 수 있다.
- 숫자 검증은 구현됐지만, 한국어로 번역된 모든 인물명·기관명·인과관계를 자동 검증하는 완전한 Fact Checker는 아니다.
- 실제 운영 모델로 전환하려면 사람이 검수한 section-level target이나 별도의 품질 평가 데이터가 필요하다.

## 7. 팀원 확인 요청사항

병합 전에 다음 사항을 함께 결정해야 한다.

1. Transformer에 상위 Paper 4개, News 2개를 전달하는 제한을 유지할지 확인
2. Finalizer의 근거 없는 수치 재작성으로 API 호출이 최대 한 번 늘어나는 정책 승인
3. 숫자 검증 실패 시 결과 전체를 차단할지, 해당 문장만 제거할지 결정
4. Neo4j 관계 타입 파라미터 필터 변경 검토
5. 안전한 근거 추출 초안을 임시 운영 방식으로 사용할지 결정
6. 새 Transformer 학습 데이터를 사람이 검수할 담당자와 평가 기준 결정

팀원 영역 변경을 원하지 않는 경우 위 파일들은 별도 커밋으로 분리하거나 되돌릴 수 있다. 박수암 담당 변경과 팀원 연동 변경을 한 커밋에 섞지 않는 것을 권장한다.

## 8. 팀 채팅 공유용 요약

> Transformer 문장 깨짐을 실제 Orchestrator E2E로 확인했습니다. 원인은 현재 모델이 예전 전체 초안 규격으로 학습됐지만 최신 코드는 섹션별 생성 규격을 사용해 모델/코드 계약이 맞지 않는 점과, 많은 RAG 결과가 384-token 입력에 동시에 들어가는 점이었습니다. 현재는 모델 계약을 검사하고, 불일치하거나 출력이 깨지면 상위 RAG 근거에서 안전한 초안을 구성하도록 보강했습니다. Transformer에는 Paper 4개·News 2개만 전달하고 LLM2/LLM3에는 전체 근거를 유지합니다. Finalizer가 근거에 없는 숫자를 만들면 한 번 재작성하고 반복되면 차단하는 테스트도 추가했습니다. 전체 테스트는 97개 통과했고 실제 한국어 E2E도 성공했습니다. 다만 `orchestrator.py`, `evidence.py`, `finalizer.py`, Graph retriever에도 연동 변경이 있으므로 병합 전에 확인 부탁드립니다. 모델 자체 품질은 아직 충분하지 않아 재학습 후보를 운영 모델로 승격하지 않았습니다.
