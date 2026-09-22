// src/App.jsx
import { useEffect, useState } from 'react';
import Intro from './components/Intro';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import GenerateForm from './components/GenerateForm';
import LoadingState from './components/LoadingState';
import EmptyState from './components/EmptyState';
import DraftResult from './components/DraftResult';
import ErrorState from './components/ErrorState';
import ScrollTopButton from './components/ScrollTopButton';
import { generateDraft } from './api/generate';
import './App.css';

const HISTORY_KEY = 'thesis-draft-history-v1';
const HISTORY_LIMIT = 20;

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

// 개발 중에는 주소 끝에 ?skipIntro 를 붙이면 인트로를 건너뜁니다.
function shouldShowIntro() {
  return !new URLSearchParams(window.location.search).has('skipIntro');
}

// status: idle -> loading -> completed | abstained | error
export default function App() {
  const [showIntro, setShowIntro] = useState(shouldShowIntro);
  const [status, setStatus] = useState('idle');
  const [draft, setDraft] = useState(null);
  const [message, setMessage] = useState('');
  const [lastInput, setLastInput] = useState(null);
  const [form, setForm] = useState({ title: '', topic: '' });
  const [history, setHistory] = useState(loadHistory);
  const [activeId, setActiveId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch {
      // 저장 공간이 없거나 막혀 있어도 화면 동작에는 영향 없음
    }
  }, [history]);

  async function handleSubmit(titleKo, topicKo) {
    setLastInput({ titleKo, topicKo });
    setStatus('loading');
    setMessage('');
    setActiveId(null);

    try {
      const result = await generateDraft(titleKo, topicKo);

      if (result.status === 'completed' && result.draft) {
        const entry = {
          id: Date.now(),
          title: titleKo,
          topic: topicKo,
          createdAt: new Date().toISOString(),
          draft: result.draft,
        };
        setDraft(result.draft);
        setActiveId(entry.id);
        setHistory((prev) => [entry, ...prev].slice(0, HISTORY_LIMIT));
        setStatus('completed');
      } else if (result.status === 'completed') {
        // 명세상 없어야 하는 응답: completed인데 draft가 없음
        setMessage('서버 응답에 초안이 없습니다. 잠시 후 다시 시도해주세요.');
        setStatus('error');
      } else {
        setMessage(result.message || '조건에 맞는 근거를 찾지 못해 생성을 보류했습니다.');
        setStatus('abstained');
      }
    } catch (err) {
      setStatus('error');
      setMessage(
        err.kind === 'input'
          ? '입력값을 다시 확인해주세요.'
          : '일시적인 서버 오류입니다. 잠시 후 다시 시도해주세요.'
      );
    }
  }

  function handleRetry() {
    if (lastInput) {
      handleSubmit(lastInput.titleKo, lastInput.topicKo);
    }
  }

  function handleNew() {
    setStatus('idle');
    setDraft(null);
    setMessage('');
    setActiveId(null);
    setForm({ title: '', topic: '' });
    setSidebarOpen(false);
  }

  function handleSelectHistory(entry) {
    setDraft(entry.draft);
    setStatus('completed');
    setMessage('');
    setActiveId(entry.id);
    setForm({ title: entry.title, topic: entry.topic || '' });
    setSidebarOpen(false);
  }

  function handleDeleteHistory(id) {
    setHistory((prev) => prev.filter((entry) => entry.id !== id));
    if (id === activeId) {
      setStatus('idle');
      setDraft(null);
      setActiveId(null);
    }
  }

  function handleClearHistory() {
    if (window.confirm('최근 기록을 모두 삭제할까요?')) {
      setHistory([]);
      if (activeId !== null) {
        setStatus('idle');
        setDraft(null);
        setActiveId(null);
      }
    }
  }

  // 모든 훅 호출이 끝난 뒤에 분기해야 합니다.
  if (showIntro) {
    return <Intro onStart={() => setShowIntro(false)} />;
  }

  const isLoading = status === 'loading';

  return (
    <div className="app-shell">
      <Sidebar
        history={history}
        activeId={activeId}
        isOpen={sidebarOpen}
        disabled={isLoading}
        onClose={() => setSidebarOpen(false)}
        onNew={handleNew}
        onSelect={handleSelectHistory}
        onDelete={handleDeleteHistory}
        onClearAll={handleClearHistory}
      />

      <div className="app-main">
        <TopBar status={status} onMenu={() => setSidebarOpen(true)} />

        <div className="workspace">
          <div className="page-intro">
            <h1>논문 초안 생성</h1>
            <p>제목과 주제를 입력하면 근거 자료를 바탕으로 서론·본론·결론 초안을 작성합니다.</p>
          </div>

          <div className="app-layout">
            <section className="app-panel-left">
              <GenerateForm
                title={form.title}
                topic={form.topic}
                onTitleChange={(title) => setForm((f) => ({ ...f, title }))}
                onTopicChange={(topic) => setForm((f) => ({ ...f, topic }))}
                onSubmit={handleSubmit}
                isLoading={isLoading}
              />
            </section>

            <main className="app-panel-right">
              {status === 'idle' && <EmptyState />}
              {status === 'loading' && <LoadingState />}
              {status === 'completed' && draft && <DraftResult draft={draft} />}
              {(status === 'abstained' || status === 'error') && (
                <ErrorState status={status} message={message} onRetry={handleRetry} />
              )}
            </main>
          </div>
        </div>
      </div>

      <ScrollTopButton />
    </div>
  );
}