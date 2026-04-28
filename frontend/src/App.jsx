import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import styles from './App.module.css'


const API = import.meta.env.VITE_API_URL || '/api'

// ── API helpers ──────────────────────────────────────────────────────────────
async function apiResolve(query) {
  const res = await fetch(`${API}/resolve?query=${encodeURIComponent(query)}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

async function apiAnalyze(ticker, query) {
  const res = await fetch(`${API}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ticker, query }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(err.detail || 'Analysis failed')
  }
  return res.json()
}

// ── Sub-components ───────────────────────────────────────────────────────────

function GraphHealthPanel({ health }) {
  return (
    <div className={styles.healthGrid}>
      <HealthCard label="Nodes" value={health.nodes} color="accent" />
      <HealthCard label="Financial Edges" value={health.financial_edges} color="blue" />
      <HealthCard label="News Edges" value={health.news_edges} color="amber" />
    </div>
  )
}

function HealthCard({ label, value, color }) {
  return (
    <div className={`${styles.healthCard} ${styles[`healthCard--${color}`]}`}>
      <span className={styles.healthValue}>{value}</span>
      <span className={styles.healthLabel}>{label}</span>
    </div>
  )
}

function EvidencePanel({ title, prefix, items, emptyMsg }) {
  const [open, setOpen] = useState(true)
  return (
    <div className={styles.panel}>
      <button className={styles.panelHeader} onClick={() => setOpen(o => !o)}>
        <span className={styles.panelTitle}>{title}</span>
        <span className={`${styles.chevron} ${open ? styles.open : ''}`}>›</span>
      </button>
      {open && (
        <div className={styles.panelBody}>
          {items.length ? items.map((item, i) => (
            <div key={i} className={styles.evidenceItem}>
              <span className={styles.evidenceIdx}>{String(i + 1).padStart(2, '0')}</span>
              <span className={styles.evidenceText}>{item}</span>
            </div>
          )) : (
            <p className={styles.empty}>{emptyMsg}</p>
          )}
        </div>
      )}
    </div>
  )
}

function PeerCard({ data }) {
  const name = data.split(' (')[0]
  const simRaw = data.split('Sim: ')[1]?.replace(')', '') ?? '0'
  const pct = (parseFloat(simRaw) * 100).toFixed(1)
  const width = `${Math.min(parseFloat(pct), 100)}%`
  return (
    <div className={styles.peerCard}>
      <div className={styles.peerTop}>
        <span className={styles.peerTicker}>{name}</span>
        <span className={styles.peerPct}>{pct}%</span>
      </div>
      <div className={styles.peerBar}>
        <div className={styles.peerFill} style={{ width }} />
      </div>
      <span className={styles.peerSub}>cosine similarity · GNN latent space</span>
    </div>
  )
}

function Spinner() {
  return (
    <div className={styles.spinnerWrap}>
      <div className={styles.spinner} />
    </div>
  )
}

function StatusLog({ messages }) {
  const ref = useRef(null)
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [messages])
  return (
    <div className={styles.statusLog} ref={ref}>
      {messages.map((m, i) => (
        <div key={i} className={styles.statusLine}>
          <span className={styles.statusTs}>{m.ts}</span>
          <span className={`${styles.statusMsg} ${styles[`status--${m.type}`]}`}>{m.text}</span>
        </div>
      ))}
      <div className={styles.cursor}>█</div>
    </div>
  )
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [query, setQuery] = useState('')
  const [phase, setPhase] = useState('idle') // idle | resolving | analyzing | done | error
  const [resolved, setResolved] = useState(null)   // { ticker, display_name }
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [log, setLog] = useState([])

  function ts() {
    return new Date().toLocaleTimeString('en-US', { hour12: false })
  }
  function addLog(text, type = 'info') {
    setLog(prev => [...prev, { ts: ts(), text, type }])
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!query.trim()) return

    setPhase('resolving')
    setResult(null)
    setResolved(null)
    setError(null)
    setLog([])

    try {
      addLog(`Resolving ticker for: "${query}"`, 'info')
      const { ticker, display_name } = await apiResolve(query)

      if (!ticker) {
        setError('Could not identify a ticker for that query. Try something like "Why is Apple falling?" or use a ticker directly like "TSLA".')
        setPhase('error')
        addLog('Ticker resolution failed.', 'error')
        return
      }

      setResolved({ ticker, display_name })
      addLog(`Resolved → ${display_name} (${ticker})`, 'success')
      setPhase('analyzing')

      addLog('Initializing coordinator agent…', 'info')
      addLog('Building Knowledge Graph…', 'info')
      addLog('Injecting real-time news sentiment…', 'info')
      addLog('Running GNN forward pass…', 'info')
      addLog('Cross-encoder reranking documents…', 'info')
      addLog('Generating causal report…', 'info')

      const data = await apiAnalyze(ticker, query)
      setResult(data)
      setPhase('done')
      addLog(`Analysis complete — ${data.graph_health.nodes} nodes in graph.`, 'success')
    } catch (err) {
      setError(err.message)
      setPhase('error')
      addLog(`Error: ${err.message}`, 'error')
    }
  }

  const busy = phase === 'resolving' || phase === 'analyzing'

  return (
    <div className={styles.shell}>
      {/* ── Header ── */}
      <header className={styles.header}>
        <div className={styles.headerInner}>
          <div className={styles.logo}>
            <span className={styles.logoMark}>L</span>
            <span className={styles.logoText}>ookUP</span>
          </div>
          <div className={styles.headerMeta}>
            <span className={styles.badge}>GNN · RAG · LLM</span>
            <span className={styles.version}>v1.0</span>
          </div>
        </div>
        <p className={styles.tagline}>
          Multi-agent causal financial reasoning &mdash; GNN Attention · Vector RAG · Gemini
        </p>
      </header>

      <main className={styles.main}>
        {/* ── Search ── */}
        <section className={styles.searchSection}>
          <form onSubmit={handleSubmit} className={styles.searchForm}>
            <div className={styles.searchWrap}>
              <span className={styles.searchPrompt}>$&gt;</span>
              <input
                className={styles.searchInput}
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder="Why has the price of gold fallen?"
                disabled={busy}
                autoFocus
              />
              {busy && <Spinner />}
            </div>
            <button
              type="submit"
              className={styles.searchBtn}
              disabled={busy || !query.trim()}
            >
              {busy ? 'Running…' : 'Analyze'}
            </button>
          </form>
          <div className={styles.hints}>
            {['Why is AAPL dropping?', 'Gold price movement', 'TSLA earnings impact', 'Infosys outlook'].map(h => (
              <button
                key={h}
                className={styles.hint}
                onClick={() => setQuery(h)}
                disabled={busy}
              >
                {h}
              </button>
            ))}
          </div>
        </section>

        {/* ── Status log ── */}
        {log.length > 0 && (
          <section className={`${styles.section} fade-up`}>
            <div className={styles.sectionLabel}>execution log</div>
            <StatusLog messages={log} />
          </section>
        )}

        {/* ── Error ── */}
        {phase === 'error' && error && (
          <section className={`${styles.section} fade-up`}>
            <div className={styles.errorBox}>
              <span className={styles.errorIcon}>⚠</span>
              <span>{error}</span>
            </div>
          </section>
        )}

        {/* ── Results ── */}
        {phase === 'done' && result && (
          <>
            {/* Resolved ticker banner */}
            <section className={`${styles.section} fade-up`} style={{ animationDelay: '0.05s' }}>
              <div className={styles.resolvedBanner}>
                <div>
                  <div className={styles.resolvedName}>{result.display_name}</div>
                  <div className={styles.resolvedTicker}>{result.ticker}</div>
                </div>
                <GraphHealthPanel health={result.graph_health} />
              </div>
            </section>

            {/* Final report */}
            <section className={`${styles.section} fade-up`} style={{ animationDelay: '0.1s' }}>
              <div className={styles.sectionLabel}>causal analysis report</div>
              <div className={styles.reportBox}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.final_report}</ReactMarkdown>
              </div>
            </section>

            {/* GNN + RAG panels */}
            <section className={`${styles.twoCol} fade-up`} style={{ animationDelay: '0.15s' }}>
              <EvidencePanel
                title="GNN Causal Drivers — Attention Mapping"
                prefix="driver"
                items={result.gnn_evidence}
                emptyMsg="No high-influence drivers identified in the current graph state."
              />
              <EvidencePanel
                title="Vector RAG — Top Reranked Documents"
                prefix="doc"
                items={result.reranked_documents}
                emptyMsg="No relevant documents retrieved."
              />
            </section>

            {/* Competitor peers */}
            {result.competitor_candidates.length > 0 && (
              <section className={`${styles.section} fade-up`} style={{ animationDelay: '0.2s' }}>
                <div className={styles.sectionLabel}>peer proximity — GNN latent space</div>
                <div className={styles.peerGrid}>
                  {result.competitor_candidates.map((c, i) => (
                    <PeerCard key={i} data={c} />
                  ))}
                </div>
              </section>
            )}
          </>
        )}

        {/* ── Idle state ── */}
        {phase === 'idle' && (
          <div className={`${styles.idleBox} fade-up`}>
            <div className={styles.idleGrid}>
              <IdleCard icon="🧠" title="GNN Attention" desc="Graph neural network maps causal relationships between news events and price movements." />
              <IdleCard icon="📑" title="Vector RAG" desc="Cross-encoder reranking surfaces the most contextually relevant documents from the knowledge store." />
              <IdleCard icon="🏢" title="Peer Proximity" desc="GNN latent-space cosine similarity identifies mathematically similar companies." />
              <IdleCard icon="✦" title="Gemini Synthesis" desc="Gemini generates a concise, data-driven causal report grounded in GNN evidence." />
            </div>
          </div>
        )}
      </main>

      <footer className={styles.footer}>
        <span>LookUP v1.0</span>
        <span className={styles.footerSep}>·</span>
        <span>Multi-Agent Graph RAG</span>
        <span className={styles.footerSep}>·</span>
        <span>GNN + Vector Store + Gemini</span>
      </footer>
    </div>
  )
}

function IdleCard({ icon, title, desc }) {
  return (
    <div className={styles.idleCard}>
      <span className={styles.idleIcon}>{icon}</span>
      <strong className={styles.idleTitle}>{title}</strong>
      <p className={styles.idleDesc}>{desc}</p>
    </div>
  )
}
