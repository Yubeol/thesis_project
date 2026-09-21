// src/components/LoadingState.jsx
import { useEffect, useState } from 'react';

const STEPS = ['근거 자료 검색', '초안 작성', '근거 검토'];

// 서버가 진행 단계를 알려주지 않으므로 경과 시간으로 추정해서 보여줍니다.
// 실서버 생성 시간을 재 본 뒤 이 기준(초)을 조정하세요.
function stepFromElapsed(seconds) {
  if (seconds < 8) return 0;
  if (seconds < 30) return 1;
  return 2;
}

export default function LoadingState() {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  const current = stepFromElapsed(elapsed);

  return (
    <div className="loading-state" aria-live="polite">
      <div className="spinner" aria-hidden="true" />
      <p className="loading-title">초안을 만들고 있습니다</p>

      <ol className="loading-steps">
        {STEPS.map((label, i) => {
          const state = i < current ? 'is-done' : i === current ? 'is-active' : '';
          return (
            <li key={label} className={`loading-step ${state}`}>
              <span className="step-dot">{i < current ? '✓' : i + 1}</span>
              {label}
            </li>
          );
        })}
      </ol>

      <p className="loading-elapsed">경과 {elapsed}초</p>
    </div>
  );
}