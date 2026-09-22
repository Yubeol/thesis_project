// src/components/LoadingState.jsx
import { useEffect, useState } from 'react';
import PipelineSteps from './PipelineSteps';
import { REQUEST_TIMEOUT_MS } from '../api/generate';
import './StateScreens.css';

const STEPS = [
  { label: '근거 자료 검색', icon: 'search', desc: '질문과 비슷한 논문·뉴스를 찾고 있어요' },
  { label: '초안 작성', icon: 'pen', desc: 'Transformer 모델이 초안을 쓰고 있어요' },
  { label: '구조 정리', icon: 'layers', desc: '서론·본론·결론으로 다듬고 근거와 맞추고 있어요' },
  { label: '도표 생성', icon: 'chart', desc: '근거 원문의 수치로 도표를 만들고 있어요' },
];

const TIPS = [
  '근거 자료 원문에 실제로 있는 수치만 도표로 만듭니다.',
  '근거가 부족하면 억지로 쓰지 않고 생성을 보류합니다.',
  '쟁점을 좁게 잡을수록 근거 검색이 정확해집니다.',
  '완성된 초안은 사이드바의 최근 기록에 자동 저장됩니다.',
];

// 원고지 위에 써지는 줄들의 길이(%)
const WRITING_LINES = [92, 100, 84, 96, 70, 100, 88, 62];

// 이 시간(초)이 지나면 "평소보다 오래 걸린다"는 안내를 추가로 보여줍니다.
const SLOW_NOTICE_SECONDS = 90;

// 타임아웃(ms)을 분 단위로 환산한 값. 안내 문구의 "최대 N분"에 사용
const MAX_MINUTES = Math.round(REQUEST_TIMEOUT_MS / 60000);

// 서버가 진행 단계를 알려주지 않으므로 경과 시간으로 추정해서 보여줍니다.
// 실모델 연결 후 E2E 생성 시간을 다시 재서 이 기준(초)을 조정하세요. (현재 잠정값)
function stepFromElapsed(seconds) {
  if (seconds < 10) return 0;
  if (seconds < 45) return 1;
  if (seconds < 70) return 2;
  return 3;
}

function formatElapsed(seconds) {
  if (seconds < 60) return `${seconds}초`;
  return `${Math.floor(seconds / 60)}분 ${seconds % 60}초`;
}

export default function LoadingState() {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  const current = stepFromElapsed(elapsed);
  const tipIndex = Math.floor(elapsed / 5) % TIPS.length;
  const isSlow = elapsed >= SLOW_NOTICE_SECONDS;

  return (
    <div className="state-screen loading-screen">
      <div className="state-head" aria-live="polite">
        <p className="state-eyebrow">
          STEP {current + 1} / {STEPS.length}
        </p>
        <h2 className="state-title">초안을 만들고 있습니다</h2>
        <p className="state-desc" key={current}>
          {STEPS[current].desc}
        </p>
      </div>

      <PipelineSteps steps={STEPS} current={current} variant="live" />

      <div className="writing-desk" aria-hidden="true">
        <div className="writing-paper">
          <span className="writing-title" />
          {WRITING_LINES.map((width, i) => (
            <span
              key={i}
              className="writing-line"
              style={{ '--i': i, '--w': `${width}%` }}
            />
          ))}
        </div>
      </div>

      <p className="loading-tip" key={tipIndex}>
        {TIPS[tipIndex]}
      </p>

      <p className="loading-foot">
        경과 {formatElapsed(elapsed)} · 단계 표시는 경과 시간 기준 추정입니다
      </p>

      {isSlow && (
        <p className="loading-foot">
          모델이 초안을 쓰는 중이라 평소보다 오래 걸리고 있어요. 최대 {MAX_MINUTES}분까지 걸릴 수 있으니 창을 닫지 말고 기다려 주세요.
        </p>
      )}
    </div>
  );
}