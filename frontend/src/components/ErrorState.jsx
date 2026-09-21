// src/components/ErrorState.jsx
export default function ErrorState({ status, message, onRetry }) {
  const isAbstained = status === 'abstained';

  return (
    <div className={`status-panel ${status}`} role="alert">
      <p>{message}</p>
      {!isAbstained && (
        <button className="retry-button" onClick={onRetry}>
          다시 시도
        </button>
      )}
    </div>
  );
}