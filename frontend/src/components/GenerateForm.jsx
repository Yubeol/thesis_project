// src/components/GenerateForm.jsx
import { useState } from 'react';

const EXAMPLES = [
  'K-POP 팬덤 문화와 2차 창작 저작권 쟁점',
  'AI 커버곡과 아티스트 퍼블리시티권',
  'K-콘텐츠 OTT 글로벌 유통과 저작권 분쟁',
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
          placeholder="예: K-POP 팬덤 문화와 2차 창작 저작권 쟁점"
        />
        {titleError && <p className="field-error">{titleError}</p>}

        <div className="example-list">
          <span className="example-label">예시</span>
          {EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              className="example-chip"
              onClick={() => handleTitleChange(example)}
              disabled={isLoading}
            >
              {example}
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