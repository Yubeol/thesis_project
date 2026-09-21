// src/components/Intro.jsx
import { useEffect, useRef, useState } from 'react';
import { SparkIcon } from './Icons';
import './Intro.css';

const COLORS = ['#7c6cff', '#f08ad8', '#4fb6f0', '#a99bff'];
const FLOW = ['RAG로 찾고', 'Transformer로 쓰고', 'LLM으로 정리'];
const EXIT_MS = 650;

export default function Intro({ onStart }) {
  const canvasRef = useRef(null);
  const exitTimer = useRef(null);
  const [leaving, setLeaving] = useState(false);

  // 반짝임 입자 (2D canvas)
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    let width = 0;
    let height = 0;
    let frame = 0;
    let particles = [];
    const mouse = { x: 0, y: 0 };

    function makeParticle(initial) {
      return {
        x: Math.random() * width,
        y: initial ? Math.random() * height : height + 20,
        size: 2 + Math.random() * 7,
        speed: 0.15 + Math.random() * 0.5,
        drift: (Math.random() - 0.5) * 0.3,
        phase: Math.random() * Math.PI * 2,
        twinkle: 0.02 + Math.random() * 0.03,
        color: COLORS[Math.floor(Math.random() * COLORS.length)],
        isStar: Math.random() < 0.55,
      };
    }

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const count = Math.min(90, Math.max(40, Math.round(width / 16)));
      particles = Array.from({ length: count }, () => makeParticle(true));
    }

    function drawStar(x, y, r) {
      ctx.beginPath();
      ctx.moveTo(x, y - r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.quadraticCurveTo(x, y, x, y + r);
      ctx.quadraticCurveTo(x, y, x - r, y);
      ctx.quadraticCurveTo(x, y, x, y - r);
      ctx.fill();
    }

    function draw() {
      ctx.clearRect(0, 0, width, height);

      for (const p of particles) {
        p.y -= p.speed;
        p.x += p.drift;
        p.phase += p.twinkle;

        if (p.y < -20) Object.assign(p, makeParticle(false));

        const alpha = 0.3 + 0.45 * (0.5 + 0.5 * Math.sin(p.phase));
        const px = p.x + mouse.x * p.size * 10;
        const py = p.y + mouse.y * p.size * 6;

        ctx.globalAlpha = alpha;
        ctx.fillStyle = p.color;

        if (p.isStar) {
          drawStar(px, py, p.size);
        } else {
          ctx.beginPath();
          ctx.arc(px, py, p.size * 0.35, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.globalAlpha = 1;
    }

    function loop() {
      draw();
      frame = requestAnimationFrame(loop);
    }

    function handleMove(e) {
      mouse.x = e.clientX / window.innerWidth - 0.5;
      mouse.y = e.clientY / window.innerHeight - 0.5;
    }

    resize();
    window.addEventListener('resize', resize);

    if (reduceMotion) {
      draw(); // 움직임 줄이기 설정: 정지 화면 한 장만 그림
    } else {
      window.addEventListener('mousemove', handleMove);
      loop();
    }

    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', handleMove);
    };
  }, []);

  useEffect(() => () => clearTimeout(exitTimer.current), []);

  function handleStart() {
    if (leaving) return;
    setLeaving(true);
    exitTimer.current = setTimeout(onStart, EXIT_MS);
  }

  return (
    <div className={`intro ${leaving ? 'is-leaving' : ''}`}>
      <div className="intro-blob intro-blob--a" />
      <div className="intro-blob intro-blob--b" />
      <div className="intro-blob intro-blob--c" />

      <div className="intro-beam" style={{ '--r': '-16deg' }} />
      <div className="intro-beam intro-beam--sky" style={{ '--r': '16deg' }} />

      <canvas ref={canvasRef} className="intro-canvas" aria-hidden="true" />

      <div className="intro-content">
        <div className="intro-logo rise" style={{ '--d': '0.1s' }}>
          <SparkIcon size={26} />
        </div>

        <div className="intro-tag rise" style={{ '--d': '0.3s' }}>
          <span className="eq" aria-hidden="true">
            <i />
            <i />
            <i />
            <i />
            <i />
          </span>
          연예 · 문화 논문 초안
        </div>

        <h1 className="intro-title rise" style={{ '--d': '0.5s' }}>
          논문 초안을 쓰는
          <br />
          <span className="intro-title-accent">AI-Agent</span>
        </h1>

        <p className="intro-desc rise" style={{ '--d': '0.7s' }}>
          제목과 주제만 넣으면 근거를 찾아 서론·본론·결론 초안을 작성합니다.
        </p>

        <ul className="intro-flow rise" style={{ '--d': '0.9s' }}>
          {FLOW.map((label, i) => (
            <li key={label}>
              <span className="intro-flow-pill">{label}</span>
              {i < FLOW.length - 1 && <span className="intro-flow-arrow">→</span>}
            </li>
          ))}
        </ul>

        <p className="intro-ask rise" style={{ '--d': '1.15s' }}>
          논문 초안 작성을 시작하시겠습니까?
        </p>

        <button
          type="button"
          className="intro-start rise"
          style={{ '--d': '1.3s' }}
          onClick={handleStart}
          autoFocus
        >
          시작하기
        </button>
      </div>
    </div>
  );
}