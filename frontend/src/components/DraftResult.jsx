// src/components/DraftResult.jsx
import { useEffect, useRef, useState } from 'react';
import { CopyIcon, DownloadIcon, CheckIcon } from './Icons';
import VisualsSection from './VisualsSection';
import './DraftResult.css';

// 다운로드/복사 파일 안에 들어가는 팀 정보. 팀명이 바뀌면 이 한 줄만 고치세요.
const TEAM_NAME = 'Team C (류민규, 박수암, 이혜림)';

const SOURCE_TYPE_LABEL = {
  paper: '논문',
  news: '뉴스',
};

const FOOTER_NOTICE = `※ 본 문서는 ${TEAM_NAME}의 논문 초안 AI-Agent(RAG + Transformer)가 생성한 초안입니다. 근거 자료를 직접 확인한 뒤 사용하세요.`;

// 논문 용지에 표시할 섹션 순서
const SECTIONS = [
  { key: 'introduction', num: 1, label: '서론' },
  { key: 'body', num: 2, label: '본론' },
  { key: 'conclusion', num: 3, label: '결론' },
];

// http/https 주소만 링크로 허용합니다. (목업의 '#'은 텍스트로만 표시됨)
function safeUrl(url) {
  return typeof url === 'string' && /^https?:\/\//i.test(url) ? url : null;
}

// LLM이 \n으로 나눈 문단을 논문 문단(<p>) 단위로 쪼갬
function toParagraphs(text) {
  if (typeof text !== 'string') return [];
  return text
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function formatDate(iso) {
  if (!iso) return null;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleDateString('ko-KR', { year: 'numeric', month: 'long', day: 'numeric' });
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
  // 시각화 카드의 출처 링크도 참고문헌과 같은 규칙(http/https만 링크)을 따르게 함
  const safeSources = sources.map((source) => ({ ...source, url: safeUrl(source.url) }));
  const dateText = formatDate(draft.generatedAt);
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
      {/* 상단 도구줄: 상태 표시 + 복사/다운로드 */}
      <div className="draft-toolbar">
        <div className="draft-toolbar-info">
          <span className="draft-toolbar-label">논문 초안 미리보기</span>
          <span className="draft-meta">
            본문 {charCount.toLocaleString('ko-KR')}자 · 근거 {sources.length}개
          </span>
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

      {/* 책상 위의 논문 용지 */}
      <div className="paper-desk">
        <article className="paper">
          <p className="paper-kicker">연예 · 문화 분야 연구 초안</p>
          <h1 className="paper-title">{draft.title}</h1>
          <p className="paper-authors">{TEAM_NAME}</p>
          {dateText && <p className="paper-date">{dateText}</p>}

          <hr className="paper-rule" />

          {SECTIONS.map((section, index) => (
            <div key={section.key}>
              <section
                className="paper-section"
                style={{ animationDelay: `${200 + index * 180}ms` }}
              >
                <h2>
                  <span className="paper-section-num">{section.num}.</span>
                  {section.label}
                </h2>
                {toParagraphs(draft[section.key]).map((paragraph, i) => (
                  <p key={i}>{paragraph}</p>
                ))}
              </section>

              {/* 본론 바로 뒤에 도표처럼 배치. visuals가 비어 있으면 섹션 자체가 숨겨짐 */}
              {section.key === 'body' && (
                <VisualsSection visuals={visuals} sources={safeSources} />
              )}
            </div>
          ))}

          {sources.length > 0 && (
            <section className="paper-references">
              <h2>참고문헌</h2>
              <ol>
                {sources.map((source, i) => {
                  const url = safeUrl(source.url);
                  return (
                    <li key={`${source.url}-${i}`} className="paper-ref">
                      <span className="paper-ref-num">[{i + 1}]</span>
                      <span className={`source-badge source-badge--${source.type}`}>
                        {SOURCE_TYPE_LABEL[source.type] ?? '기타'}
                      </span>
                      {url ? (
                        <a href={url} target="_blank" rel="noreferrer">
                          {source.title}
                        </a>
                      ) : (
                        <span className="paper-ref-title">{source.title}</span>
                      )}
                    </li>
                  );
                })}
              </ol>
            </section>
          )}

          <footer className="paper-footer">
            {FOOTER_NOTICE}
            <span className="paper-page">- 1 -</span>
          </footer>
        </article>
      </div>
    </div>
  );
}