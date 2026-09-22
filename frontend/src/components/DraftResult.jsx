// src/components/DraftResult.jsx
import { useEffect, useRef, useState } from 'react';
import { CopyIcon, DownloadIcon, CheckIcon } from './Icons';
import VisualsSection from './VisualsSection';

// 다운로드/복사 파일 안에 들어가는 팀 정보. 팀명이 바뀌면 이 한 줄만 고치세요.
const TEAM_NAME = 'Team C (류민규, 박수암, 이혜림)';

const SOURCE_TYPE_LABEL = {
  paper: '논문',
  news: '뉴스',
};

const FOOTER_NOTICE = `※ 본 문서는 ${TEAM_NAME}의 논문 초안 AI-Agent(RAG + Transformer)가 생성한 초안입니다. 근거 자료를 직접 확인한 뒤 사용하세요.`;

// http/https 주소만 링크로 허용합니다. (목업의 '#'은 텍스트로만 표시됨)
function safeUrl(url) {
  return typeof url === 'string' && /^https?:\/\//i.test(url) ? url : null;
}

function sourceLine(source, i) {
  const label = SOURCE_TYPE_LABEL[source.type] ?? '기타';
  const url = safeUrl(source.url);
  return `${i + 1}. [${label}] ${source.title}${url ? ` — ${url}` : ''}`;
}

function buildMarkdown(draft) {
  const lines = [
    `# ${draft.title}`,
    '',
    `**팀명:** ${TEAM_NAME}`,
    '',
    '## 서론',
    draft.introduction,
    '',
    '## 본론',
    draft.body,
    '',
    '## 결론',
    draft.conclusion,
  ];

  if (draft.sources && draft.sources.length > 0) {
    lines.push('', '## 근거 자료');
    draft.sources.forEach((source, i) => lines.push(sourceLine(source, i)));
  }

  lines.push('', '---', FOOTER_NOTICE);
  return lines.join('\n');
}

// 마크다운 기호 없이, 메모장이나 한글에 붙여 넣기 좋은 순수 텍스트
function buildPlainText(draft) {
  const lines = [
    draft.title,
    `팀명: ${TEAM_NAME}`,
    '',
    '[서론]',
    draft.introduction,
    '',
    '[본론]',
    draft.body,
    '',
    '[결론]',
    draft.conclusion,
  ];

  if (draft.sources && draft.sources.length > 0) {
    lines.push('', '[근거 자료]');
    draft.sources.forEach((source, i) => lines.push(sourceLine(source, i)));
  }

  lines.push('', '----------------', FOOTER_NOTICE);
  return lines.join('\n');
}

// 형식별 설정. Word(.docx)는 백엔드 API 명세가 나오면 여기에 추가합니다.
const DOWNLOAD_FORMATS = [
  {
    id: 'txt',
    label: '텍스트 (.txt)',
    hint: '메모장·한글에 붙여넣기 좋음',
    ext: 'txt',
    mime: 'text/plain;charset=utf-8',
    build: buildPlainText,
  },
  {
    id: 'md',
    label: '마크다운 (.md)',
    hint: 'GitHub·노션용',
    ext: 'md',
    mime: 'text/markdown;charset=utf-8',
    build: buildMarkdown,
  },
];

export default function DraftResult({ draft }) {
  const [copyState, setCopyState] = useState('idle'); // idle | done | fail
  const [menuOpen, setMenuOpen] = useState(false);
  const resetTimer = useRef(null);
  const menuRef = useRef(null);

  useEffect(() => () => clearTimeout(resetTimer.current), []);

  // 메뉴 바깥을 누르거나 Esc를 누르면 닫기
  useEffect(() => {
    if (!menuOpen) return undefined;

    function handlePointer(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setMenuOpen(false);
      }
    }
    function handleKey(e) {
      if (e.key === 'Escape') setMenuOpen(false);
    }

    document.addEventListener('mousedown', handlePointer);
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('mousedown', handlePointer);
      document.removeEventListener('keydown', handleKey);
    };
  }, [menuOpen]);

  const sources = draft.sources ?? [];
  const visuals = draft.visuals ?? [];
  // 시각화 카드의 출처 링크도 출처 목록과 같은 규칙(http/https만 링크)을 따르게 함
  const safeSources = sources.map((source) => ({ ...source, url: safeUrl(source.url) }));
  const charCount =
    (draft.introduction?.length ?? 0) +
    (draft.body?.length ?? 0) +
    (draft.conclusion?.length ?? 0);

  function flashCopyState(next) {
    setCopyState(next);
    clearTimeout(resetTimer.current);
    resetTimer.current = setTimeout(() => setCopyState('idle'), 2000);
  }

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(buildMarkdown(draft));
      flashCopyState('done');
    } catch {
      flashCopyState('fail');
    }
  }

  function handleDownload(format) {
    const blob = new Blob([format.build(draft)], { type: format.mime });
    const href = URL.createObjectURL(blob);
    const safeName =
      draft.title.replace(/[\\/:*?"<>|]/g, '').trim().slice(0, 60) || '논문초안';

    const link = document.createElement('a');
    link.href = href;
    link.download = `${safeName}.${format.ext}`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(href);

    setMenuOpen(false);
  }

  return (
    <div className="draft-result">
      <div className="draft-header">
        <div>
          <h2 className="draft-title">{draft.title}</h2>
          <p className="draft-meta">
            본문 {charCount.toLocaleString('ko-KR')}자 · 근거 {sources.length}개
          </p>
        </div>

        <div className="draft-actions">
          <button type="button" className="action-button" onClick={handleCopy}>
            {copyState === 'done' ? <CheckIcon size={15} /> : <CopyIcon size={15} />}
            {copyState === 'done' ? '복사됨' : copyState === 'fail' ? '복사 실패' : '복사'}
          </button>

          <div className="download-wrap" ref={menuRef}>
            <button
              type="button"
              className="action-button"
              onClick={() => setMenuOpen((open) => !open)}
              aria-haspopup="menu"
              aria-expanded={menuOpen}
            >
              <DownloadIcon size={15} />
              다운로드
            </button>

            {menuOpen && (
              <ul className="download-menu" role="menu">
                {DOWNLOAD_FORMATS.map((format) => (
                  <li key={format.id} role="none">
                    <button
                      type="button"
                      role="menuitem"
                      className="download-item"
                      onClick={() => handleDownload(format)}
                    >
                      <span className="download-item-label">{format.label}</span>
                      <span className="download-item-hint">{format.hint}</span>
                    </button>
                  </li>
                ))}
                <li role="none">
                  <button type="button" role="menuitem" className="download-item" disabled>
                    <span className="download-item-label">Word (.docx)</span>
                    <span className="download-item-hint">준비 중</span>
                  </button>
                </li>
              </ul>
            )}
          </div>
        </div>
      </div>

      <article className="draft-block">
        <h3>서론</h3>
        <p>{draft.introduction}</p>
      </article>

      <article className="draft-block">
        <h3>본론</h3>
        <p>{draft.body}</p>
      </article>

      <article className="draft-block">
        <h3>결론</h3>
        <p>{draft.conclusion}</p>
      </article>

      {/* 근거 자료 원문에서 뽑은 차트·표. visuals가 비어 있으면 섹션 자체가 숨겨짐 */}
      <VisualsSection visuals={visuals} sources={safeSources} />

      {sources.length > 0 && (
        <aside className="draft-sources">
          <h4>근거 자료</h4>
          <ul>
            {sources.map((source, i) => {
              const url = safeUrl(source.url);
              return (
                <li key={`${source.url}-${i}`} className="source-item">
                  <span className="source-index">{i + 1}</span>
                  <span className={`source-badge source-badge--${source.type}`}>
                    {SOURCE_TYPE_LABEL[source.type] ?? '기타'}
                  </span>
                  {url ? (
                    <a href={url} target="_blank" rel="noreferrer">
                      {source.title}
                    </a>
                  ) : (
                    <span className="source-title-plain">{source.title}</span>
                  )}
                </li>
              );
            })}
          </ul>
        </aside>
      )}
    </div>
  );
}