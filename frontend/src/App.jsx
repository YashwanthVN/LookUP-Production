import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import styles from './App.module.css'


const API = import.meta.env.DEV ? '/api' : (import.meta.env.VITE_API_URL || '/api')

const ABSTAIN_TEXT = {
  no_news_for_primary_ticker:
    'No recent headlines about this company were found, so no explanation is offered. Peer headlines are shown as context only.',
  no_timestamped_news_in_window:
    'No timestamped news was found for this company or its peers in the last 72 hours, so no explanation is offered.',
}

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

function formatWhen(iso) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })
}

// ── Sub-components ───────────────────────────────────────────────────────────

function CoveragePanel({ diag, asOf }) {
  const d = diag || {}
  return (
    <div>
      <div className={styles.healthGrid}>
        <HealthCard label="Headlines (72h)" value={d.pool_size ?? 0} color="accent" />
        <HealthCard label="About this company" value={d.own_count ?? 0} color="blue" />
      </div>
      {asOf && <div className={styles.peerSub}>as of {formatWhen(asOf)}</div>}
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

function HeadlinePanel({ title, items, emptyMsg }) {
  const [open, setOpen] = useState(true)
  return (
    <div className={styles.panel}>
      <button className={styles.panelHeader} onClick={() => setOpen(o => !o)}>
        <span className={styles.panelTitle}>{title}</span>
        <span className={`${styles.chevron} ${open ? styles.open : ''}`}>›</span>
      </button>
      {open && (
        <div className={styles.panelBody}>
          {items.length ? items.map((it, i) => (
            <div key={i} className={styles.evidenceItem}>
              <span className={styles.evidenceIdx}>{String(i + 1).padStart(2, '0')}</span>
              <span className={styles.evidenceText}>
                {it.url ? (
                  <a href={it.url} target="_blank" rel="noopener noreferrer">{it.headline}</a>
                ) : it.headline}
                <br />
                <span className={styles.peerSub}>
                  {it.is_primary ? 'this company' : `peer: ${it.ticker}`}
                  {' · '}{it.source || 'unknown source'}
                  {' · '}{formatWhen(it.published_at)}
                  {' · '}relevance {it.rerank_score.toFixed(2)}
                </span>
              </span>
            </div>
          )) : (
            <p className={styles.empty}>{emptyMsg}</p>
          )}
        </div>
      )}
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

      addLog('Collecting news from the last 72 hours (company and sector peers)…', 'info')
      addLog('Ranking headlines against your question…', 'info')

      const data = await apiAnalyze(ticker, query)
      setResult(data)
      setPhase('done')
      const n = data.diagnostics?.pool_size ?? 0
      if (data.abstained) {
        addLog(`No explanation offered (${data.abstain_reason}). ${n} headlines considered.`, 'error')
      } else {
        addLog(`Done — ranked ${n} headlines.`, 'success')
      }
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
            <span className={styles.badge}>NEWS · CROSS-ENCODER</span>
            <span className={styles.version}>v1.1</span>
          </div>
        </div>
        <p className={styles.tagline}>
          Recent news ranked by relevance to your question &mdash; and an honest &ldquo;no answer&rdquo; when there isn&rsquo;t enough evidence
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
            <section className={`${styles.section} fade-up`} style={{ animationDelay: '0.05s' }}>
              <div className={styles.resolvedBanner}>
                <div>
                  <div className={styles.resolvedName}>{result.display_name}</div>
                  <div className={styles.resolvedTicker}>{result.ticker}</div>
                </div>
                <CoveragePanel diag={result.diagnostics} asOf={result.as_of} />
              </div>
            </section>

            {result.abstained && (
              <section className={`${styles.section} fade-up`} style={{ animationDelay: '0.08s' }}>
                <div className={styles.errorBox}>
                  <span className={styles.errorIcon}>ⓘ</span>
                  <span>{ABSTAIN_TEXT[result.abstain_reason] || 'Not enough evidence to offer an explanation.'}</span>
                </div>
              </section>
            )}

            <section className={`${styles.section} fade-up`} style={{ animationDelay: '0.1s' }}>
              <div className={styles.sectionLabel}>{result.abstained ? 'context' : 'relevant headlines'}</div>
              <div className={styles.reportBox}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.final_report}</ReactMarkdown>
              </div>
            </section>

            <section className={`${styles.section} fade-up`} style={{ animationDelay: '0.15s' }}>
              <HeadlinePanel
                title="Sources — top-ranked headlines"
                items={result.ranked || []}
                emptyMsg="No headlines to show."
              />
            </section>
          </>
        )}

        {/* ── Idle state ── */}
        {phase === 'idle' && (
          <div className={`${styles.idleBox} fade-up`}>
            <div className={styles.idleGrid}>
              <IdleCard icon="🕒" title="Point-in-time news" desc="Only headlines published in the 72 hours before your question, each with its publish time and source." />
              <IdleCard icon="📑" title="Cross-encoder ranking" desc="A cross-encoder scores every headline against your question and surfaces the most relevant." />
              <IdleCard icon="🏢" title="Sector context" desc="Peer-company headlines are shown as context, and labelled as such." />
              <IdleCard icon="✋" title="Says when it doesn't know" desc="If there is no news about the company itself, LookUP says so instead of guessing." />
            </div>
          </div>
        )}
      </main>

      <footer className={styles.footer}>
        <span>LookUP v1.1</span>
        <span className={styles.footerSep}>·</span>
        <span>A relevance ranking, not a verified explanation of price moves</span>
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