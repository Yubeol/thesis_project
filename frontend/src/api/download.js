// src/api/download.js
//
// 백엔드에서 파일(PDF, Word)을 만들어 받아오는 요청 모음.
// 두 형식 모두 같은 데이터를 보내고, 주소와 응답 형식만 다릅니다.
//   POST /api/download/pdf   -> application/pdf
//   POST /api/download/docx  -> application/vnd.openxmlformats-officedocument.wordprocessingml.document
// 도표(visuals)는 아직 파일에 포함되지 않습니다.

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// 서버가 파일을 만드는 데 멈춰 있으면 버튼이 계속 "생성 중"으로 남으므로 제한을 둡니다. (ms)
const FILE_TIMEOUT_MS = 60000;

// 초안 -> 백엔드 요청 형식. url이 없으면 백엔드 문서 생성에서 오류가 날 수 있어 빈 문자열로 보냅니다.
function toPayload(draft) {
  const { title, introduction, body, conclusion, sources = [] } = draft;
  return {
    draft: { title, introduction, body, conclusion },
    sources: sources.map(({ type, title: sourceTitle, url }) => ({
      type,
      title: sourceTitle,
      url: typeof url === 'string' ? url : '',
    })),
  };
}

async function requestFile(path, draft, expectedType) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FILE_TIMEOUT_MS);

  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(toPayload(draft)),
      signal: controller.signal,
    });
  } catch (cause) {
    const error = new Error(`file-download-network-error${path}`);
    error.cause = cause;
    throw error;
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok || !response.headers.get('content-type')?.includes(expectedType)) {
    throw new Error(`file-download-failed-${response.status}${path}`);
  }

  return response.blob();
}

export function downloadPdf(draft) {
  return requestFile('/api/download/pdf', draft, 'application/pdf');
}

export function downloadDocx(draft) {
  return requestFile('/api/download/docx', draft, 'wordprocessingml');
}