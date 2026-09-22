// src/components/GenerateForm.jsx
import { useState } from 'react';

// 예시 칩: 누르면 제목과 주제를 함께 채웁니다.
// 첫 번째 예시는 근거에 수치가 있어 도표(visuals)가 나오는 주제입니다. (혜림님 제공 테스트 입력)
// 나머지 저작권 쟁점 주제는 수치가 적어 도표 없이 초안만 나오는 게 정상입니다.
const EXAMPLES = [
  {
    title: 'K-pop의 글로벌 확산에서 디지털 플랫폼의 역할',
    topic:
      'TikTok, YouTube, Instagram 등 디지털 플랫폼이 K-pop의 글로벌 확산과 팬덤 형성에 미친 영향을 분석',
  },
  {
    title: 'K-POP 팬덤 문화와 2차 창작 저작권 쟁점',
    topic: '팬아트·팬픽·커버 영상 등 팬덤의 2차 창작이 원저작권과 충돌하는 지점과 허용 범위를 분석',
  },
  {
    title: 'AI 커버곡과 아티스트 퍼블리시티권',
    topic: 'AI로 가수의 목소리를 복제한 커버곡이 아티스트의 퍼블리시티권을 침해하는지 쟁점을 정리',
  },
];

export default function GenerateForm({
  title,
  topic,
  onTitleChange,
  onTopicChange,
  onSubmit,
  isLoading,
}) {
  const [titleError, setTitleError] = useState('');

  function handleTitleChange(value) {
    if (titleError) setTitleError('');
    onTitleChange(value);
  }

  function handleExample(example) {
    handleTitleChange(example.title);
    onTopicChange(example.topic);
  }

  function handleSubmit(e) {
    e.preventDefault();

    const trimmedTitle = title.trim();
    if (!trimmedTitle) {
      setTitleError('논문 제목을 입력해주세요.');
      return;
    }
    setTitleError('');
    onSubmit(trimmedTitle, topic.trim() || null);
  }

  return (
    <form onSubmit={handleSubmit} className="generate-form">
      <h2 className="panel-title">초안 만들기</h2>
      <p className="panel-desc">
        쟁점을 좁게 쓸수록 근거 검색이 정확해집니다.
      </p>

      <div className="field">
        <label htmlFor="title-ko">
          논문 제목 <span className="required">*</span>
        </label>
        <input
          id="title-ko"
          type="text"
          value={title}
          onChange={(e) => handleTitleChange(e.target.value)}
          disabled={isLoading}
          placeholder="분석할 쟁점이 드러나는 제목을 입력하세요"
        />
        {titleError && <p className="field-error">{titleError}</p>}

        <div className="example-list">
          <span className="example-label">예시</span>
          {EXAMPLES.map((example) => (
            <button
              key={example.title}
              type="button"
              className="example-chip"
              onClick={() => handleExample(example)}
              disabled={isLoading}
              title={example.topic}
            >
              {example.title}
            </button>
          ))}
        </div>
      </div>

      <div className="field">
        <label htmlFor="topic-ko">주제 설명 (선택)</label>
        <textarea
          id="topic-ko"
          value={topic}
          onChange={(e) => onTopicChange(e.target.value)}
          disabled={isLoading}
          placeholder="추가로 참고할 방향이 있다면 적어주세요"
          rows={5}
        />
        <p className="field-count">{topic.length}자</p>
      </div>

      <button type="submit" className="submit-button" disabled={isLoading}>
        {isLoading ? '생성 중...' : '초안 생성'}
      </button>
    </form>
  );
}