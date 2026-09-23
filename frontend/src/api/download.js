const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function downloadPdf(draft) {
  const {
    title,
    introduction,
    body,
    conclusion,
    sources = [],
    visuals = [],
    generatedAt = null,
  } = draft;
  const response = await fetch(`${API_BASE_URL}/api/download/pdf`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      draft: { title, introduction, body, conclusion },
      sources: sources.map(({ type, title: sourceTitle, url }) => ({
        type,
        title: sourceTitle,
        url,
      })),
      visuals,
      generated_at: generatedAt,
    }),
  });

  if (!response.ok || !response.headers.get('content-type')?.includes('application/pdf')) {
    throw new Error(`pdf-download-failed-${response.status}`);
  }

  return response.blob();
}
