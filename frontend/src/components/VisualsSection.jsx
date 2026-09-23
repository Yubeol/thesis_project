import { useId } from 'react'
import './VisualsSection.css'

const PALETTE = ['#6d5efc', '#38bdf8', '#f472b6', '#a78bfa', '#34d399', '#fbbf24']
const KIND_LABEL = { line: '추이', bar: '비교', pie: '비율', table: '표' }

// 카드 실제 폭(약 300~600px)에 가깝게 잡아야 축 글자가 과하게 축소되지 않는다.
const W = 420
const H = 250

// [수정] 항목이 많으면 가로축 라벨을 기울이고, 그만큼 아래 여백과 높이를 늘린다.
const ROTATE_AFTER = 6
const ROTATED_EXTRA = 34

function chartFrame(rotated) {
  const height = H + (rotated ? ROTATED_EXTRA : 0)
  return {
    height,
    plot: { left: 46, right: W - 12, top: 18, bottom: height - (rotated ? 70 : 36) },
  }
}

// [수정] 순위 차트 판정: 숫자가 작을수록 좋은 데이터
const RANK_UNIT = /^(위|순위|등)$/
const RANK_TITLE = /순위|rank/i

const color = (i) => PALETTE[i % PALETTE.length]

function toNumber(value) {
  const n = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(n) ? n : null
}

function formatNumber(n) {
  return Number(n).toLocaleString('ko-KR', { maximumFractionDigits: 2 })
}

function percent(value, total) {
  return `${((value / total) * 100).toFixed(1).replace(/\.0$/, '')}%`
}

function truncate(text, max) {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text
}

function valueText(value, unit, rank) {
  return rank ? `${formatNumber(value)}위` : `${formatNumber(value)}${unit}`
}

// 표이거나 항목이 많은 그래프는 한 줄 전체를 차지한다.
function isWide(visual) {
  return visual.kind === 'table' || (visual.labels && visual.labels.length > 8)
}

// 단위가 '위'/'순위'이거나 제목에 순위·rank가 있고 값이 모두 1 이상 정수면 순위 차트로 본다.
function detectRank(kind, title, unit, series) {
  if (kind !== 'bar' && kind !== 'line') return false
  if (!RANK_UNIT.test(unit.trim()) && !RANK_TITLE.test(title)) return false
  return series.every((s) => s.values.every((v) => Number.isInteger(v) && v >= 1))
}

// 백엔드(LLM)가 만든 데이터를 검사해서 그릴 수 있는 형태로 정리한다.
// 형식이 어긋난 항목은 null을 반환해 화면에서 제외한다.
function normalizeVisual(raw) {
  if (!raw || typeof raw !== 'object') return null

  const kind = raw.kind
  const title =
    typeof raw.title === 'string' && raw.title.trim() ? raw.title.trim() : '제목 없는 시각화'
  const sourceIndex = Number.isInteger(raw.source_index) ? raw.source_index : null
  const unit = typeof raw.unit === 'string' ? raw.unit : ''

  if (kind === 'table') {
    const columns = Array.isArray(raw.columns) ? raw.columns.map(String) : []
    const rows = Array.isArray(raw.rows)
      ? raw.rows
          .filter(Array.isArray)
          .map((row) => columns.map((_, i) => (row[i] == null ? '' : String(row[i]))))
      : []
    if (!columns.length || !rows.length) return null
    return { kind, title, sourceIndex, unit, columns, rows, rank: false }
  }

  if (!['line', 'bar', 'pie'].includes(kind)) return null

  const labels = Array.isArray(raw.labels) ? raw.labels.map(String) : []
  if (!labels.length) return null

  const series = (Array.isArray(raw.series) ? raw.series : [])
    .map((s, i) => {
      const values = Array.isArray(s?.values) ? s.values.map(toNumber) : []
      if (values.length !== labels.length || values.some((v) => v === null)) return null
      const name = typeof s?.name === 'string' && s.name.trim() ? s.name.trim() : `계열 ${i + 1}`
      return { name, values }
    })
    .filter(Boolean)
  if (!series.length) return null

  if (kind === 'pie') {
    const values = series[0].values
    const total = values.reduce((a, b) => a + b, 0)
    if (values.some((v) => v < 0) || total <= 0) return null
    return { kind, title, sourceIndex, unit, labels, series: [series[0]], rank: false }
  }

  const rank = detectRank(kind, title, unit, series)
  return { kind, title, sourceIndex, unit, labels, series, rank }
}

// 축 눈금을 1, 2, 2.5, 5 단위의 보기 좋은 값으로 맞춘다.
function niceScale(minValue, maxValue, ticks = 4) {
  const lo = Math.min(0, minValue)
  const hi = Math.max(0, maxValue)
  if (lo === hi) return { min: 0, max: 1, step: 0.25 }
  const rawStep = (hi - lo) / ticks
  const magnitude = 10 ** Math.floor(Math.log10(rawStep))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= rawStep)
  return { min: Math.floor(lo / step) * step, max: Math.ceil(hi / step) * step, step }
}

// 일반 수치 축: 아래가 작은 값
function valueAxis(values) {
  const { min, max, step } = niceScale(Math.min(...values), Math.max(...values))
  const count = Math.round((max - min) / step)
  const ticks = Array.from({ length: count + 1 }, (_, i) => Number((min + step * i).toFixed(10)))
  return { min, max, ticks, inverted: false }
}

// 순위 축: 위가 1위. 최하위 막대가 사라지지 않도록 끝에 한 칸 여유를 둔다.
function rankAxis(values) {
  const worst = Math.max(...values)
  const raw = Math.max(worst, 2) / 4
  const magnitude = 10 ** Math.floor(Math.log10(raw))
  let step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw)
  step = Math.max(1, Math.ceil(step))
  let max = Math.ceil(worst / step) * step
  if (max <= worst) max += step
  const ticks = [1]
  for (let v = step; v <= max; v += step) if (v > 1) ticks.push(v)
  return { min: 1, max, ticks, inverted: true }
}

function makeY(axis, plot) {
  const span = axis.max - axis.min
  const h = plot.bottom - plot.top
  return axis.inverted
    ? (v) => plot.top + ((v - axis.min) / span) * h
    : (v) => plot.bottom - ((v - axis.min) / span) * h
}

function YGrid({ axis, y, plot, rank }) {
  return (
    <g>
      {axis.ticks.map((v, i) => (
        <g key={i}>
          <line className="viz-grid" x1={plot.left} x2={plot.right} y1={y(v)} y2={y(v)} />
          <text
            className="viz-axis-text"
            x={plot.left - 8}
            y={y(v)}
            textAnchor="end"
            dominantBaseline="middle"
          >
            {rank ? `${formatNumber(v)}위` : formatNumber(v)}
          </text>
        </g>
      ))}
    </g>
  )
}

function XLabel({ x, label, plot, rotated }) {
  if (rotated) {
    const ly = plot.bottom + 14
    return (
      <text
        className="viz-axis-text"
        x={x}
        y={ly}
        textAnchor="end"
        transform={`rotate(-35 ${x} ${ly})`}
      >
        <title>{label}</title>
        {truncate(label, 12)}
      </text>
    )
  }
  return (
    <text className="viz-axis-text" x={x} y={plot.bottom + 22} textAnchor="middle">
      <title>{label}</title>
      {truncate(label, 7)}
    </text>
  )
}

function BarChart({ visual }) {
  const { labels, series, unit, rank } = visual
  const rotated = labels.length > ROTATE_AFTER
  const { height, plot } = chartFrame(rotated)
  const all = series.flatMap((s) => s.values)
  const axis = rank ? rankAxis(all) : valueAxis(all)
  const y = makeY(axis, plot)
  // 순위 차트는 아래 기준선에서 위로 자라므로 1위 막대가 가장 길다.
  const baseY = rank ? plot.bottom : y(0)
  const band = (plot.right - plot.left) / labels.length
  const groupWidth = band * 0.68
  const barWidth = groupWidth / series.length
  const showValues = series.length === 1 && labels.length <= (rank ? 14 : 6)
  const labelStep = labels.length > 16 ? Math.ceil(labels.length / 16) : 1

  return (
    <svg className="viz-svg" viewBox={`0 0 ${W} ${height}`} role="img" aria-label={visual.title}>
      <YGrid axis={axis} y={y} plot={plot} rank={rank} />
      {labels.map((label, i) => {
        const groupX = plot.left + band * i + (band - groupWidth) / 2
        return (
          <g key={i}>
            {series.map((s, si) => {
              const v = s.values[i]
              const top = Math.min(y(v), baseY)
              const barHeight = Math.max(Math.abs(baseY - y(v)), 0.5)
              return (
                <rect
                  key={si}
                  className="viz-bar"
                  x={groupX + barWidth * si + 1}
                  y={top}
                  width={Math.max(barWidth - 2, 1)}
                  height={barHeight}
                  rx={Math.min(5, barWidth / 3)}
                  fill={color(si)}
                  style={{ animationDelay: `${i * 70 + si * 40}ms` }}
                >
                  <title>{`${label} · ${s.name}: ${valueText(v, unit, rank)}`}</title>
                </rect>
              )
            })}
            {showValues && (
              <text
                className="viz-value-text"
                x={groupX + groupWidth / 2}
                y={Math.min(y(series[0].values[i]), baseY) - 6}
                textAnchor="middle"
                style={{ animationDelay: `${i * 70 + 400}ms` }}
              >
                {valueText(series[0].values[i], '', rank)}
              </text>
            )}
            {i % labelStep === 0 && (
              <XLabel x={groupX + groupWidth / 2} label={label} plot={plot} rotated={rotated} />
            )}
          </g>
        )
      })}
      <line className="viz-baseline" x1={plot.left} x2={plot.right} y1={baseY} y2={baseY} />
    </svg>
  )
}

function LineChart({ visual }) {
  const { labels, series, unit, rank } = visual
  const gradientId = `viz-area-${useId().replace(/[^a-zA-Z0-9-]/g, '')}`
  const n = labels.length
  const rotated = n > ROTATE_AFTER
  const { height, plot } = chartFrame(rotated)
  const all = series.flatMap((s) => s.values)
  const axis = rank ? rankAxis(all) : valueAxis(all)
  const y = makeY(axis, plot)
  const inner = 14
  const x = (i) =>
    n === 1
      ? (plot.left + plot.right) / 2
      : plot.left + inner + ((plot.right - plot.left - inner * 2) * i) / (n - 1)
  const labelStep = n > 16 ? Math.ceil(n / 16) : 1
  const pathOf = (values) => values.map((v, i) => `${i ? 'L' : 'M'}${x(i)},${y(v)}`).join(' ')

  return (
    <svg className="viz-svg" viewBox={`0 0 ${W} ${height}`} role="img" aria-label={visual.title}>
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color(0)} stopOpacity="0.28" />
          <stop offset="100%" stopColor={color(0)} stopOpacity="0" />
        </linearGradient>
      </defs>
      <YGrid axis={axis} y={y} plot={plot} rank={rank} />
      {series.length === 1 && n > 1 && (
        <path
          className="viz-area"
          d={`${pathOf(series[0].values)} L${x(n - 1)},${plot.bottom} L${x(0)},${plot.bottom} Z`}
          fill={`url(#${gradientId})`}
        />
      )}
      {series.map((s, si) => (
        <g key={si}>
          {n > 1 && (
            <path
              className="viz-line"
              d={pathOf(s.values)}
              pathLength="1"
              stroke={color(si)}
              style={{ animationDelay: `${si * 150}ms` }}
            />
          )}
          {s.values.map((v, i) => (
            <circle
              key={i}
              className="viz-dot"
              cx={x(i)}
              cy={y(v)}
              r="4"
              stroke={color(si)}
              style={{ animationDelay: `${700 + i * 60}ms` }}
            >
              <title>{`${labels[i]} · ${s.name}: ${valueText(v, unit, rank)}`}</title>
            </circle>
          ))}
        </g>
      ))}
      {labels.map((label, i) =>
        i % labelStep === 0 ? (
          <XLabel key={i} x={x(i)} label={label} plot={plot} rotated={rotated} />
        ) : null,
      )}
    </svg>
  )
}

function PieChart({ visual }) {
  const { labels, unit } = visual
  const values = visual.series[0].values
  const total = values.reduce((a, b) => a + b, 0)
  const R = 70
  const STROKE = 34
  const C = 2 * Math.PI * R
  const gap = values.filter((v) => v > 0).length > 1 ? 1.5 : 0
  const topIndex = values.indexOf(Math.max(...values))
  // 단위가 이미 %면 비율을 한 번 더 붙이지 않는다. (45% · 45% 중복 방지)
  const legendValue = (v) =>
    unit === '%' ? `${formatNumber(v)}%` : `${formatNumber(v)}${unit} · ${percent(v, total)}`

  let offset = 0
  const slices = values.map((v, i) => {
    const length = (v / total) * C
    const slice = { i, length, offset }
    offset += length
    return slice
  })

  return (
    <div className="viz-pie-wrap">
      <svg className="viz-pie" viewBox="0 0 200 200" role="img" aria-label={visual.title}>
        <circle className="viz-pie-track" cx="100" cy="100" r={R} strokeWidth={STROKE} />
        {slices.map(({ i, length, offset: start }) =>
          length > 0 ? (
            <circle
              key={i}
              className="viz-slice"
              cx="100"
              cy="100"
              r={R}
              stroke={color(i)}
              strokeWidth={STROKE}
              strokeDasharray={`${Math.max(length - gap, 0.1)} ${C}`}
              strokeDashoffset={-start}
              transform="rotate(-90 100 100)"
              style={{ animationDelay: `${i * 90}ms` }}
            >
              <title>{`${labels[i]}: ${legendValue(values[i])}`}</title>
            </circle>
          ) : null,
        )}
        <text className="viz-pie-center-value" x="100" y="98" textAnchor="middle">
          {percent(values[topIndex], total)}
        </text>
        <text className="viz-pie-center-label" x="100" y="118" textAnchor="middle">
          {truncate(labels[topIndex], 8)}
        </text>
      </svg>
      <ul className="viz-legend viz-legend--column">
        {labels.map((label, i) => (
          <li key={i}>
            <span className="viz-swatch" style={{ background: color(i) }} />
            <span className="viz-legend-label">{label}</span>
            <span className="viz-legend-value">{legendValue(values[i])}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function DataTable({ visual }) {
  return (
    <div className="viz-table-wrap">
      <table className="viz-table">
        <thead>
          <tr>
            {visual.columns.map((col, i) => (
              <th key={i} scope="col">
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {visual.rows.map((row, ri) => (
            <tr key={ri}>
              {row.map((cell, ci) => (
                <td key={ci}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function SeriesLegend({ series }) {
  return (
    <ul className="viz-legend">
      {series.map((s, i) => (
        <li key={i}>
          <span className="viz-swatch" style={{ background: color(i) }} />
          {s.name}
        </li>
      ))}
    </ul>
  )
}

function VisualCard({ visual, sources, index, wide }) {
  const source = visual.sourceIndex !== null ? sources[visual.sourceIndex] : null

  return (
    <article
      className={`viz-card${wide ? ' viz-card--wide' : ''}`}
      style={{ animationDelay: `${index * 120}ms` }}
    >
      <header className="viz-card-head">
        <span className={`viz-kind viz-kind--${visual.kind}`}>{KIND_LABEL[visual.kind]}</span>
        <h4 className="viz-title">{visual.title}</h4>
        {visual.rank ? (
          <span className="viz-unit">순위 · 위쪽이 상위</span>
        ) : (
          visual.unit &&
          visual.kind !== 'table' && <span className="viz-unit">단위: {visual.unit}</span>
        )}
      </header>

      {visual.kind === 'bar' && <BarChart visual={visual} />}
      {visual.kind === 'line' && <LineChart visual={visual} />}
      {visual.kind === 'pie' && <PieChart visual={visual} />}
      {visual.kind === 'table' && <DataTable visual={visual} />}

      {(visual.kind === 'bar' || visual.kind === 'line') && visual.series.length > 1 && (
        <SeriesLegend series={visual.series} />
      )}

      <footer className="viz-source">
        {source ? (
          <>
            <span className="viz-source-tag">
              근거 {visual.sourceIndex + 1} · {source.type === 'paper' ? '논문' : '뉴스'}
            </span>
            {source.url ? (
              <a href={source.url} target="_blank" rel="noopener noreferrer" title={source.title}>
                {source.title}
              </a>
            ) : (
              <span className="viz-source-title">{source.title}</span>
            )}
          </>
        ) : (
          <span>근거 자료 번호가 지정되지 않은 시각화입니다.</span>
        )}
      </footer>
    </article>
  )
}

// 2열 그리드에서 좁은 카드가 홀수 개로 끝나 빈칸이 생기면 마지막 카드를 한 줄로 펼친다.
// 넓은 카드(표 등)가 사이에 끼면 그 앞뒤 구간을 따로 센다.
function layoutWidths(items) {
  const widths = items.map(isWide)
  let run = []
  const flush = () => {
    if (run.length % 2 === 1) widths[run[run.length - 1]] = true
    run = []
  }
  items.forEach((_, i) => {
    if (widths[i]) flush()
    else run.push(i)
  })
  flush()
  return widths
}

export default function VisualsSection({ visuals, sources = [] }) {
  const items = (Array.isArray(visuals) ? visuals : []).map(normalizeVisual).filter(Boolean)
  if (!items.length) return null

  const widths = layoutWidths(items)

  return (
    <section className="visuals-section" aria-label="근거 자료 시각화">
      <div className="visuals-header">
        <h3>근거 자료 시각화</h3>
        <p>검색된 근거 자료 원문에 있는 수치와 내용만 사용해 만들었습니다.</p>
      </div>
      <div className="visuals-grid">
        {items.map((visual, i) => (
          <VisualCard key={i} visual={visual} sources={sources} index={i} wide={widths[i]} />
        ))}
      </div>
    </section>
  )
}