// src/components/DraftResult.jsx
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { CopyIcon, DownloadIcon, CheckIcon } from './Icons';
import VisualsSection from './VisualsSection';
import { downloadPdf, downloadDocx } from '../api/download';
import './DraftResult.css';

// 다운로드/복사 파일 안에 들어가는 팀 정보. 팀명이 바뀌면 이 한 줄만 고치세요.
const TEAM_NAME = 'Team C (류민규, 박수암, 이혜림)';

// 논문 첫 장 제목 위, 2쪽부터 머리글 오른쪽에 들어가는 분야 표시
const FIELD_LABEL = '연예 · 문화 분야 연구 초안';

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

// A4 세로 비율(297 / 210). 종이 폭에 곱해서 한 장의 높이를 구함
const PAGE_RATIO = 1.4142;
// 2쪽부터 맨 위에 붙는 머리글이 차지하는 높이(px). DraftResult.css의 .paper-running-head와 맞춤
const RUNNING_HEAD_SPACE = 44;

// http/https 주소만 링크로 허용합니다. (목업의 '#'은 텍스트로만 표시됨)
function safeUrl(url) {
  return typeof url === 'string' && /^https?:\/\//i.test(url) ? url : null;
}

// 파일 이름에 쓸 수 없는 문자를 빼고 60자로 자름. 비면 기본 이름 사용
function makeFileName(title) {
  return (title ?? '').replace(/[\\/:*?"<>|]/g, '').trim().slice(0, 60) || '논문초안';
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

// 초안을 페이지에 나눠 담을 수 있는 작은 블록 단위로 쪼갬.
// 문단·참고문헌 한 줄이 한 블록이고, 제목(keepWithNext)은 다음 블록과 같은 페이지에 붙음.
function buildBlocks(draft, hasVisuals) {
  const blocks = [{ type: 'front' }];

  SECTIONS.forEach((section) => {
    blocks.push({ type: 'heading', section, keepWithNext: true });
    toParagraphs(draft[section.key]).forEach((text) => blocks.push({ type: 'para', text }));
    if (section.key === 'body' && hasVisuals) blocks.push({ type: 'visuals' });
  });

  const sources = draft.sources ?? [];
  if (sources.length > 0) {
    blocks.push({ type: 'ref-heading', keepWithNext: true });
    sources.forEach((source, index) => blocks.push({ type: 'ref', source, index }));
  }

  blocks.push({ type: 'notice' });
  return blocks;
}

// 측정한 블록 높이(costs)를 앞에서부터 담다가, 한 장 용량을 넘으면 다음 장으로 넘김.
// 한 블록이 한 장보다 크면 자르지 않고 그 장이 길어지도록 둠.
function paginate(blocks, costs, firstCapacity, restCapacity) {
  const pages = [[]];
  let used = 0;

  blocks.forEach((block, i) => {
    const capacity = pages.length === 1 ? firstCapacity : restCapacity;
    const hasNext = i + 1 < blocks.length;
    const need = costs[i] + (block.keepWithNext && hasNext ? costs[i + 1] : 0);
    const current = pages[pages.length - 1];

    if (current.length > 0 && used + need > capacity) {
      pages.push([i]);
      used = costs[i];
    } else {
      current.push(i);
      used += costs[i];
    }
  });

  return pages;
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

// 브라우저에서 바로 저장하는 텍스트 형식
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

// 백엔드가 만들어 주는 파일 형식 (도표는 아직 포함되지 않음)
const SERVER_FORMATS = [
  {
    id: 'pdf',
    name: 'PDF',
    label: 'PDF (.pdf)',
    hint: '서론·본론·결론과 근거 자료',
    ext: 'pdf',
    request: downloadPdf,
  },
  {
    id: 'docx',
    name: 'Word',
    label: 'Word (.docx)',
    hint: '한글·Word에서 이어서 편집',
    ext: 'docx',
    request: downloadDocx,
  },
];

// Blob을 파일로 저장. revoke를 약간 늦춰야 일부 브라우저에서 다운로드가 끊기지 않음
function saveBlob(blob, fileName) {
  const href = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = href;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(href), 1000);
}

export default function DraftResult({ draft }) {
  const [copyState, setCopyState] = useState('idle'); // idle | done | fail
  const [menuOpen, setMenuOpen] = useState(false);
  const [serverBusy, setServerBusy] = useState(null); // null | 'pdf' | 'docx'
  const [serverError, setServerError] = useState(null); // null | 'pdf' | 'docx'
  const [layout, setLayout] = useState(null); // { blocks, pages }
  const [pageWidth, setPageWidth] = useState(0);
  const [fontsVersion, setFontsVersion] = useState(0);
  const resetTimer = useRef(null);
  const menuRef = useRef(null);
  const measureRef = useRef(null);

  useEffect(() => () => clearTimeout(resetTimer.current), []);

  // 다른 초안으로 바뀌면 이전 초안의 다운로드 오류 표시를 지움
  useEffect(() => {
    setServerError(null);
  }, [draft]);

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

  // 종이 폭이 바뀌면(창 크기 변경 등) 페이지를 다시 나눔
  useEffect(() => {
    const el = measureRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return undefined;
    const observer = new ResizeObserver(() => setPageWidth(el.offsetWidth));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // 명조 웹폰트가 늦게 로딩되면 글자 높이가 달라지므로 다시 측정
  useEffect(() => {
    let alive = true;
    document.fonts?.ready.then(() => {
      if (alive) setFontsVersion((v) => v + 1);
    });
    return () => {
      alive = false;
    };
  }, []);

  const sources = draft.sources ?? [];
  const visuals = draft.visuals ?? [];
  // 시각화 카드의 출처 링크도 참고문헌과 같은 규칙(http/https만 링크)을 따르게 함
  const safeSources = sources.map((source) => ({ ...source, url: safeUrl(source.url) }));
  const dateText = formatDate(draft.generatedAt);
  const charCount =
    (draft.introduction?.length ?? 0) +
    (draft.body?.length ?? 0) +
    (draft.conclusion?.length ?? 0);

  const blocks = useMemo(() => buildBlocks(draft, visuals.length > 0), [draft, visuals.length]);

  // 숨겨진 측정용 종이에서 블록 높이를 재고 페이지를 나눔.
  // useLayoutEffect라서 화면에 그려지기 전에 끝나 깜빡임이 없음.
  useLayoutEffect(() => {
    const el = measureRef.current;
    if (!el) return;
    const width = el.offsetWidth;
    if (!width) return;

    const style = window.getComputedStyle(el);
    const padTop = parseFloat(style.paddingTop) || 0;
    const padBottom = parseFloat(style.paddingBottom) || 0;
    const firstCapacity = width * PAGE_RATIO - padTop - padBottom;
    const restCapacity = firstCapacity - RUNNING_HEAD_SPACE;

    // 각 블록의 높이 = 다음 블록 시작 위치 - 이 블록 시작 위치 (여백 포함)
    const nodes = Array.from(el.children);
    const contentEnd = el.scrollHeight - padBottom;
    const costs = nodes.map((node, i) => {
      const nextTop = i + 1 < nodes.length ? nodes[i + 1].offsetTop : contentEnd;
      return Math.max(nextTop - node.offsetTop, 0);
    });

    setLayout({ blocks, pages: paginate(blocks, costs, firstCapacity, restCapacity) });
  }, [blocks, pageWidth, fontsVersion]);

  // 초안이 바뀐 직후 한 번은 이전 페이지 정보가 남아 있으므로, 같은 blocks일 때만 사용
  const pages = layout && layout.blocks === blocks ? layout.pages : null;

  const errorFormat = SERVER_FORMATS.find((format) => format.id === serverError);

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
    saveBlob(blob, `${makeFileName(draft.title)}.${format.ext}`);
    setMenuOpen(false);
  }

  // PDF·Word처럼 백엔드가 파일을 만들어 주는 형식
  async function handleServerDownload(format) {
    setMenuOpen(false);
    setServerBusy(format.id);
    setServerError(null);

    try {
      const blob = await format.request(draft);
      saveBlob(blob, `${makeFileName(draft.title)}_teamC.${format.ext}`);
    } catch {
      setServerError(format.id);
    } finally {
      setServerBusy(null);
    }
  }

  function renderBlock(block) {
    switch (block.type) {
      case 'front':
        return (
          <div className="paper-front">
            <p className="paper-kicker">{FIELD_LABEL}</p>
            <h1 className="paper-title">{draft.title}</h1>
            <p className="paper-authors">{TEAM_NAME}</p>
            {dateText && <p className="paper-date">{dateText}</p>}
            <hr className="paper-rule" />
          </div>
        );
      case 'heading':
        return (
          <h2 className="paper-heading">
            <span className="paper-section-num">{block.section.num}.</span>
            {block.section.label}
          </h2>
        );
      case 'para':
        return <p className="paper-para">{block.text}</p>;
      case 'visuals':
        return <VisualsSection visuals={visuals} sources={safeSources} />;
      case 'ref-heading':
        return <h2 className="paper-ref-heading">참고문헌</h2>;
      case 'ref': {
        const url = safeUrl(block.source.url);
        return (
          <div className="paper-ref">
            <span className="paper-ref-num">[{block.index + 1}]</span>
            <span className={`source-badge source-badge--${block.source.type}`}>
              {SOURCE_TYPE_LABEL[block.source.type] ?? '기타'}
            </span>
            {url ? (
              <a href={url} target="_blank" rel="noreferrer">
                {block.source.title}
              </a>
            ) : (
              <span className="paper-ref-title">{block.source.title}</span>
            )}
          </div>
        );
      }
      case 'notice':
        return <p className="paper-notice">{FOOTER_NOTICE}</p>;
      default:
        return null;
    }
  }

  return (
    <div className="draft-result">
      {/* 상단 도구줄: 상태 표시 + 복사/다운로드 */}
      <div className="draft-toolbar">
        <div className="draft-toolbar-info">
          <span className="draft-toolbar-label">논문 초안 미리보기</span>
          <span className="draft-meta">
            본문 {charCount.toLocaleString('ko-KR')}자 · 근거 {sources.length}개
            {pages ? ` · 총 ${pages.length}쪽` : ''}
          </span>
          {errorFormat && (
            <span className="draft-meta" role="alert">
              {errorFormat.name} 다운로드에 실패했습니다. 백엔드를 확인해주세요.
            </span>
          )}
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
              {serverBusy ? '파일 만드는 중...' : '다운로드'}
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
                {SERVER_FORMATS.map((format) => (
                  <li key={format.id} role="none">
                    <button
                      type="button"
                      role="menuitem"
                      className="download-item"
                      disabled={serverBusy !== null}
                      onClick={() => handleServerDownload(format)}
                    >
                      <span className="download-item-label">{format.label}</span>
                      <span className="download-item-hint">
                        {serverBusy === format.id ? '생성 중...' : format.hint}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>

      <div className="paper-desk">
        {/* 높이 측정용 숨은 종이: 블록 전체를 한 장에 이어서 그려 놓고 높이만 잼 */}
        <div className="paper paper-measure" ref={measureRef} aria-hidden="true">
          {blocks.map((block, i) => (
            <div key={i} className="paper-block">
              {renderBlock(block)}
            </div>
          ))}
        </div>

        {/* 실제로 보이는 A4 비율 종이들 */}
        {pages && (
          <div className="paper-stack">
            {pages.map((indexes, p) => (
              <article
                key={p}
                className="paper paper-sheet"
                style={{ animationDelay: `${Math.min(p, 4) * 120}ms` }}
              >
                {p > 0 && (
                  <div className="paper-running-head">
                    <span className="paper-running-title">{draft.title}</span>
                    <span>{FIELD_LABEL}</span>
                  </div>
                )}

                <div className="paper-body">
                  {indexes.map((i) => (
                    <div key={i} className="paper-block">
                      {renderBlock(blocks[i])}
                    </div>
                  ))}
                </div>

                <span className="paper-page-num">
                  - {p + 1} / {pages.length} -
                </span>
              </article>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}