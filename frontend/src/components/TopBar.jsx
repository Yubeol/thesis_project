// src/components/TopBar.jsx
import { MenuIcon } from './Icons';

const STATUS_LABEL = {
  idle: '대기 중',
  loading: '생성 중',
  completed: '생성 완료',
  abstained: '생성 보류',
  error: '오류',
};

export default function TopBar({ status, onMenu }) {
  return (
    <header className="topbar">
      <button
        type="button"
        className="icon-button menu-button"
        onClick={onMenu}
        aria-label="메뉴 열기"
      >
        <MenuIcon />
      </button>

      <div className="topbar-title">
        <strong>초안 생성</strong>
        <span className="topbar-tag">연예 · 문화</span>
      </div>

      <div className="topbar-spacer" />

      <span className={`status-pill status-pill--${status}`}>
        <span className="status-dot" />
        {STATUS_LABEL[status] ?? STATUS_LABEL.idle}
      </span>
    </header>
  );
}