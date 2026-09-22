// src/api/generate.js
//
// 백엔드 응답 형식:
// {
//   status: 'completed' | 'abstained',
//   draft: { title, introduction, body, conclusion } | null,
//   sources: [{ type: 'paper' | 'news', title, url }],
//   visuals: [{ kind: 'line' | 'bar' | 'pie' | 'table', title, ... , source_index }],
//   message: string | null
// }
//
// 서버 응답의 sources와 visuals는 draft 밖에 있지만, 화면 컴포넌트는 draft 하나만
// 받도록 normalizeResponse에서 draft.sources / draft.visuals로 합쳐서 넘깁니다.
// 응답 형식이 바뀌면 이 파일의 normalizeResponse만 고치면 됩니다.

import { MOCK_VISUALS } from './mockVisuals';

// 디자인 작업 중에는 목업 사용. 실서버 테스트 시 false로 변경.
const USE_MOCK = false;

// 목업일 때 재현할 상황: 'completed' | 'abstained' | 'error'
const MOCK_SCENARIO = 'completed'; // 'completed' | 'abstained' | 'error'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// 근거 재검증 단계 때문에 생성이 오래 걸릴 수 있어 넉넉하게 잡습니다. (ms)
const REQUEST_TIMEOUT_MS = 120000;

function mockDelay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function mockGenerateDraft(titleKo) {
  await mockDelay(1500);

  if (MOCK_SCENARIO === 'error') {
    const error = new Error('mock-server-error');
    error.kind = 'server';
    throw error;
  }

  if (MOCK_SCENARIO === 'abstained') {
    return {
      status: 'abstained',
      draft: null,
      sources: [],
      visuals: [],
      message: '입력하신 주제와 관련된 근거 자료를 충분히 찾지 못했습니다.',
    };
  }

  return {
    status: 'completed',
    draft: {
      title: titleKo,
      introduction: `(목업) "${titleKo}"에 대한 서론입니다. 문제와 배경을 다룹니다.\n(목업) 두 번째 문단입니다. 연구의 범위와 쟁점을 좁혀 제시합니다.`,
      body: '(목업) 찾은 근거 자료와 모델이 작성한 내용을 종합한 본론입니다.\n(목업) 근거 자료별 주장을 비교하고 쟁점을 분석하는 문단입니다.',
      conclusion: '(목업) 요약과 제안을 담은 결론입니다.',
    },
    sources: [
      { type: 'paper', title: '(목업) 참고 논문 1', url: '#' },
      { type: 'paper', title: '(목업) 참고 논문 2', url: '#' },
      { type: 'news', title: '(목업) 참고 기사 1', url: '#' },
    ],
    visuals: MOCK_VISUALS,
    message: null,
  };
}

// 서버 응답 -> 화면에서 쓰는 형태로 변환
function normalizeResponse(data) {
  const sources = Array.isArray(data.sources) ? data.sources : [];
  const visuals = Array.isArray(data.visuals) ? data.visuals : [];
  return {
    status: data.status,
    draft: data.draft
      ? {
          ...data.draft,
          sources,
          visuals,
          // 논문 용지에 표시할 생성일. 기록 복원 시에도 원래 날짜가 유지됨
          generatedAt: new Date().toISOString(),
        }
      : null,
    message: data.message ?? null,
  };
}

/**
 * @param {string} titleKo 논문 제목 (필수)
 * @param {string|null} topicKo 주제 설명 (선택)
 * @returns {Promise<{status: 'completed'|'abstained', draft: object|null, message: string|null}>}
 * @throws {Error} err.kind가 'input' 또는 'server'로 설정된 에러
 */
export async function generateDraft(titleKo, topicKo) {
  if (USE_MOCK) {
    return normalizeResponse(await mockGenerateDraft(titleKo));
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response;
  try {
    response = await fetch(`${API_BASE_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title_ko: titleKo,
        topic_ko: topicKo || null,
      }),
      signal: controller.signal,
    });
  } catch (cause) {
    // 네트워크 오류나 타임아웃은 서버 쪽 문제로 취급합니다.
    const error = new Error('generate-network-error');
    error.kind = 'server';
    error.cause = cause;
    throw error;
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    const kind = response.status >= 400 && response.status < 500 ? 'input' : 'server';
    const error = new Error(`generate-failed-${response.status}`);
    error.kind = kind;
    throw error;
  }

  return normalizeResponse(await response.json());
}