// src/components/ScrollTopButton.jsx
// 화면(뷰포트) 오른쪽 아래에 항상 고정되어 있는 "맨 위로" 버튼.
// 테두리 링이 페이지를 읽은 만큼 차오르고, 맨 위에 있을 때는 살짝 흐리게 표시됨.
import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import './ScrollTopButton.css';

const RING_R = 21;
const RING_C = 2 * Math.PI * RING_R;

// 위치 고정은 CSS 파일이 안 읽혀도 동작하도록 인라인으로 지정
const PIN_STYLE = {
  position: 'fixed',
  right: '28px',
  bottom: '28px',
  zIndex: 35,
};

export default function ScrollTopButton() {
  const [atTop, setAtTop] = useState(true);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    let frame = 0;

    function update() {
      frame = 0;
      const y = window.scrollY;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      setAtTop(y <= 4);
      setProgress(max > 0 ? Math.min(y / max, 1) : 0);
    }

    // 스크롤 이벤트마다 계산하지 않고 한 프레임에 한 번만 갱신
    function schedule() {
      if (!frame) frame = requestAnimationFrame(update);
    }

    update();
    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule);
    return () => {
      window.removeEventListener('scroll', schedule);
      window.removeEventListener('resize', schedule);
      cancelAnimationFrame(frame);
    };
  }, []);

  function handleClick() {
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    window.scrollTo({ top: 0, behavior: reduceMotion ? 'auto' : 'smooth' });
  }

  // body 바로 아래에 그려서, 부모 요소의 transform 등에 영향받지 않고 항상 화면 기준으로 고정
  return createPortal(
    <button
      type="button"
      className={`scroll-top${atTop ? ' is-at-top' : ''}`}
      style={PIN_STYLE}
      onClick={handleClick}
      aria-label="맨 위로 이동"
      title="맨 위로"
    >
      <svg className="scroll-top-ring" viewBox="0 0 48 48" aria-hidden="true">
        <circle className="scroll-top-track" cx="24" cy="24" r={RING_R} />
        <circle
          className="scroll-top-progress"
          cx="24"
          cy="24"
          r={RING_R}
          strokeDasharray={RING_C}
          strokeDashoffset={RING_C * (1 - progress)}
          transform="rotate(-90 24 24)"
        />
      </svg>
      <svg
        className="scroll-top-arrow"
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.4"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        <path d="M12 19V5" />
        <path d="m5 12 7-7 7 7" />
      </svg>
    </button>,
    document.body,
  );
}