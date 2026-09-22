// src/components/PipelineSteps.jsx
// 로딩 화면과 빈 화면이 함께 쓰는 단계 그림.
// variant="live"면 current 기준으로 진행 상태를 표시하고, "static"이면 안내용으로 멈춰 있음.
import './StateScreens.css';

const ICON_PATHS = {
  search: ['M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14Z', 'M21 21l-4.35-4.35'],
  pen: ['M12 20h9', 'M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z'],
  layers: ['M12 2 2 7l10 5 10-5-10-5Z', 'M2 17l10 5 10-5', 'M2 12l10 5 10-5'],
  chart: ['M3 3v18h18', 'M7 16v-4', 'M12 16V8', 'M17 16v-7'],
  check: ['M20 6 9 17l-5-5'],
};

function StepIcon({ name }) {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {(ICON_PATHS[name] ?? []).map((d) => (
        <path key={d} d={d} />
      ))}
    </svg>
  );
}

export default function PipelineSteps({ steps, current = -1, variant = 'live' }) {
  return (
    <ol className={`pipeline pipeline--${variant}`} style={{ '--steps': steps.length }}>
      {steps.map((step, i) => {
        let state = 'idle';
        if (variant === 'live') {
          if (i < current) state = 'done';
          else if (i === current) state = 'active';
        }

        // 노드 i와 i+1을 잇는 선: 지나온 구간은 채우고, 현재 구간은 빛이 흐름
        let linkState = '';
        if (variant === 'live') {
          if (i < current) linkState = 'is-filled';
          else if (i === current) linkState = 'is-flowing';
        }

        return (
          <li
            key={step.label}
            className={`pipeline-step is-${state}`}
            style={{ animationDelay: `${i * 120}ms` }}
          >
            <span className="pipeline-node">
              <StepIcon name={state === 'done' ? 'check' : step.icon} />
            </span>
            <span className="pipeline-label">{step.label}</span>
            {i < steps.length - 1 && (
              <span className={`pipeline-link ${linkState}`} aria-hidden="true" />
            )}
          </li>
        );
      })}
    </ol>
  );
}