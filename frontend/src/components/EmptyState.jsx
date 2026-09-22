// src/components/EmptyState.jsx
// 아직 생성한 초안이 없을 때 오른쪽 패널에 보여주는 안내 화면
import PipelineSteps from './PipelineSteps';
import './StateScreens.css';

const GUIDE_STEPS = [
  { label: '제목 입력', icon: 'pen' },
  { label: '근거 검색', icon: 'search' },
  { label: '초안 작성', icon: 'layers' },
  { label: '도표 정리', icon: 'chart' },
];

const FEATURES = [
  {
    title: '근거 연결',
    desc: '참고문헌 번호로 어떤 자료를 근거로 썼는지 보여줍니다.',
  },
  {
    title: '도표 자동 생성',
    desc: '근거 원문에 있는 수치만으로 차트와 표를 만듭니다.',
  },
  {
    title: '근거 부족 시 보류',
    desc: '근거가 모자라면 억지로 쓰지 않고 이유를 알려줍니다.',
  },
];

export default function EmptyState() {
  return (
    <div className="state-screen empty-screen">
      <div className="state-head">
        <p className="state-eyebrow">HOW IT WORKS</p>
        <h2 className="state-title">쟁점 하나를 정하면, 근거를 찾아 초안을 씁니다</h2>
        <p className="state-desc">
          왼쪽에 논문 제목을 입력하고 초안 생성을 눌러보세요. 생성한 초안은 사이드바의 최근
          기록에 저장됩니다.
        </p>
      </div>

      <PipelineSteps steps={GUIDE_STEPS} variant="static" />

      <ul className="feature-list">
        {FEATURES.map((feature, i) => (
          <li
            key={feature.title}
            className="feature-card"
            style={{ animationDelay: `${300 + i * 120}ms` }}
          >
            <strong>{feature.title}</strong>
            <span>{feature.desc}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}