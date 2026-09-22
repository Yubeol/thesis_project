// 목업 전용 시각화 데이터 (화면 확인용, 실제 수치 아님)
// USE_MOCK이 true일 때만 generate.js에서 사용한다.
export const MOCK_VISUALS = [
  {
    kind: 'line',
    title: '연도별 AI 커버곡 관련 보도 건수',
    labels: ['2021', '2022', '2023', '2024'],
    series: [{ name: '보도 건수', values: [12, 27, 58, 91] }],
    unit: '건',
    source_index: 1,
  },
  {
    kind: 'bar',
    title: '플랫폼별 AI 생성 음원 관련 신고 건수',
    labels: ['유튜브', '틱톡', '스트리밍 A', '스트리밍 B'],
    series: [
      { name: '2023년', values: [34, 21, 9, 6] },
      { name: '2024년', values: [52, 38, 17, 11] },
    ],
    unit: '건',
    source_index: 0,
  },
  {
    kind: 'pie',
    title: '근거 자료에서 다룬 쟁점 비중',
    labels: ['퍼블리시티권', '저작권', '계약·수익 배분'],
    series: [{ name: '비중', values: [45, 35, 20] }],
    unit: '%',
    source_index: 2,
  },
  {
    kind: 'table',
    title: '국가별 AI 생성 음성 규율 방식 비교',
    columns: ['국가', '규율 방식', '특징'],
    rows: [
      ['미국', '주(州) 단위 퍼블리시티권 법률', '주마다 보호 범위가 다름'],
      ['EU', 'AI 법의 투명성 의무', '생성물 표시 의무 중심'],
      ['한국', '부정경쟁방지법 퍼블리시티 조항', '성명·초상 등의 무단 사용 규율'],
    ],
    source_index: 0,
  },
]