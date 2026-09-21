// src/components/Sidebar.jsx
import {
  PlusIcon,
  ClockIcon,
  TrashIcon,
  SparkIcon,
  ShieldIcon,
  CloseIcon,
} from './Icons';

const RULES = [
  '쟁점은 하나만 좁게',
  '뉴스·웹·공개 지식만 사용',
  '근거가 초안에 보이게',
  '없는 사실·가짜 인용 금지',
];

function formatDate(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString('ko-KR', {
    month: 'numeric',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export default function Sidebar({
  history,
  activeId,
  isOpen,
  disabled,
  onClose,
  onNew,
  onSelect,
  onDelete,
  onClearAll,
}) {
  return (
    <>
      {isOpen && <div className="sidebar-overlay" onClick={onClose} />}

      <aside className={`sidebar ${isOpen ? 'is-open' : ''}`}>
        <div className="sidebar-head">
          <div className="sidebar-brand">
            <span className="brand-mark">
              <SparkIcon size={18} />
            </span>
            <div className="brand-text">
              <strong>Thesis Agent</strong>
              <span>논문 초안 AI-Agent</span>
            </div>
          </div>
          <button
            type="button"
            className="icon-button sidebar-close"
            onClick={onClose}
            aria-label="사이드바 닫기"
          >
            <CloseIcon />
          </button>
        </div>

        <button type="button" className="new-button" onClick={onNew} disabled={disabled}>
          <PlusIcon size={16} />
          새 초안 만들기
        </button>

        <section className="sidebar-section">
          <div className="sidebar-section-head">
            <h3 className="sidebar-section-title">
              <ClockIcon size={14} />
              최근 기록
            </h3>
            {history.length > 0 && (
              <button
                type="button"
                className="text-button"
                onClick={onClearAll}
                disabled={disabled}
              >
                전체 삭제
              </button>
            )}
          </div>

          {history.length === 0 ? (
            <p className="history-empty">아직 생성한 초안이 없습니다.</p>
          ) : (
            <ul className="history-list">
              {history.map((entry) => (
                <li
                  key={entry.id}
                  className={`history-item ${entry.id === activeId ? 'is-active' : ''}`}
                >
                  <button
                    type="button"
                    className="history-select"
                    onClick={() => onSelect(entry)}
                    disabled={disabled}
                  >
                    <span className="history-title">{entry.title}</span>
                    <span className="history-date">{formatDate(entry.createdAt)}</span>
                  </button>
                  <button
                    type="button"
                    className="history-delete"
                    onClick={() => onDelete(entry.id)}
                    disabled={disabled}
                    aria-label="기록 삭제"
                  >
                    <TrashIcon size={14} />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="rules-card">
          <h3 className="rules-title">
            <ShieldIcon size={15} />
            지켜야 할 4가지
          </h3>
          <ol>
            {RULES.map((rule) => (
              <li key={rule}>{rule}</li>
            ))}
          </ol>
        </section>
      </aside>
    </>
  );
}