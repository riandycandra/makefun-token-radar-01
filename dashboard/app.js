(function () {
  const D = window.RADAR;
  const root = document.getElementById('board');
  if (!D || !root) return;
  root.setAttribute('aria-busy', 'false');

  const fmtUsd = (n) => {
    if (n === null || n === undefined) return '—';
    if (n >= 1e9) return '$' + (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6) return '$' + (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3) return '$' + (n / 1e3).toFixed(1) + 'K';
    return '$' + Math.round(n);
  };
  const fmtPct = (v) => {
    if (v === null || v === undefined) return '—';
    const s = (v >= 0 ? '+' : '') + v.toFixed(1) + '%';
    return `<span class="${v >= 0 ? 'pos' : 'neg'}">${s}</span>`;
  };
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  let state = {
    view: 'board', // 'board' | 'live'
    tier: 'all', // 'all' | 'watchlist' | 'early'
    category: 'all',
    chain: 'all',
    search: '',
    selectedSlug: null,
  };

  const categories = [...new Set(D.rows.map((r) => r.category || 'Uncategorized'))].sort();
  const chains = [...new Set(D.rows.flatMap((r) => r.chains))].sort();

  function buildFarmStepsHtml(r) {
    const ch = r.chains && r.chains.length ? r.chains[0] : 'its main chain';
    const cap = r.farm_capital_usd;
    const gas = r.farm_gas_usd;
    const hold = r.farm_hold_days;
    const steps = [
      `Isolate a dedicated wallet on ${ch} funded with $${cap} USDC + $${gas} gas.`,
      `Deposit $${cap} into ${r.name} (${r.category || 'DeFi'}) on ${ch} and hold the receipt.`,
      `Maintain weekly on-chain cadence (~$${Math.max(1, Math.round(gas / 2))}/wk gas, 10m/wk).`,
      `Hold across at least ${hold} days to span multiple snapshot epochs.`,
      `Verify rewards only on ${r.name}'s official portal before withdrawing.`,
      `Log tx hashes, wallet address, and snapshot dates for sybil hygiene.`
    ];
    return steps.map((s) => `<li>${esc(s)}</li>`).join('');
  }

  function render() {
    const u = D.universe;
    const f = D.fit;
    const maxMult = Math.max(...f.category_multiples.map((m) => m.multiple), 0.1);

    const filteredRows = D.rows.filter((r) => {
      if (state.tier === 'watchlist' && !r.on_watchlist) return false;
      if (state.tier === 'early' && r.on_watchlist) return false;
      if (state.category !== 'all' && (r.category || 'Uncategorized') !== state.category) return false;
      if (state.chain !== 'all' && !r.chains.includes(state.chain)) return false;
      if (state.search) {
        const q = state.search.toLowerCase();
        if (!r.name.toLowerCase().includes(q) && !r.slug.toLowerCase().includes(q)) return false;
      }
      return true;
    });

    const selected = state.selectedSlug ? D.rows.find((r) => r.slug === state.selectedSlug) : null;

    root.innerHTML = `
      <div class="wrap">
        <header class="header">
          <div>
            <div class="title">TOKEN LAUNCH RADAR · EMPIRICAL LAUNCH VALUATION</div>
            <div class="subtitle">All ${u.total} DefiLlama protocols accounted for · As of ${esc(D.as_of)} · Model ${esc(D.model_version)}</div>
          </div>
          <div class="badge-pill">${f.folds}-Fold Out-of-Sample CV · MdAPE ${f.cv_mdape_pct}% vs ${f.baseline_mdape_pct}% Baseline</div>
        </header>

        <section class="kpis">
          <div class="kpi"><div class="kpi-label">Universe</div><div class="kpi-val">${u.total}</div><div class="kpi-sub">All protocols ≥ $5M TVL</div></div>
          <div class="kpi"><div class="kpi-label">Watchlist Floor</div><div class="kpi-val">${u.watchlist}</div><div class="kpi-sub">TVL ≥ ${fmtUsd(D.floor_usd)}</div></div>
          <div class="kpi"><div class="kpi-label">Board Scored</div><div class="kpi-val">${u.board_scored}</div><div class="kpi-sub">${D.early_size} early pre-token</div></div>
          <div class="kpi"><div class="kpi-label">Live Anchors</div><div class="kpi-val">${u.live_with_mcap}</div><div class="kpi-sub">${u.token_without_mcap} unpriced tokens</div></div>
          <div class="kpi"><div class="kpi-label">CV Spearman ρ</div><div class="kpi-val">${f.cv_spearman}</div><div class="kpi-sub">5-fold rank correlation</div></div>
          <div class="kpi"><div class="kpi-label">Global Median</div><div class="kpi-val">${f.global_median_multiple}x</div><div class="kpi-sub">MCap / TVL ratio</div></div>
        </section>

        <section class="panels">
          <div class="panel">
            <h2>Out-of-Sample 5-Fold Cross-Validation (n = ${f.live_sample_n} Live Tokens)</h2>
            <div class="cv-grid">
              <div class="cv-stat"><div class="l">CV Log R²</div><div class="v">${f.cv_r2}</div></div>
              <div class="cv-stat"><div class="l">CV Spearman</div><div class="v">${f.cv_spearman}</div></div>
              <div class="cv-stat"><div class="l">Model MdAPE</div><div class="v pos">${f.cv_mdape_pct}%</div></div>
              <div class="cv-stat"><div class="l">0.25x Baseline</div><div class="v">${f.baseline_mdape_pct}%</div></div>
            </div>
            <div class="subtitle" style="margin-top:10px;">
              Ridge weights: intercept=${f.weights.intercept}, logTVL=${f.weights.log_tvl}, logChains=${f.weights.log_chains}, maturity=${f.weights.maturity}
            </div>
          </div>
          <div class="panel">
            <h2>Empirical Shrinkage Category Multiples (MCap / TVL)</h2>
            <div class="mult-bars">
              ${f.category_multiples.slice(0, 8).map((m) => `
                <div class="mult-row">
                  <span style="min-width:115px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${esc(m.category)} (${m.n})</span>
                  <div class="mult-bar-bg"><div class="mult-bar-fill" style="width:${Math.min(100, (m.multiple / maxMult) * 100)}%"></div></div>
                  <span class="mono">${m.multiple.toFixed(2)}x</span>
                </div>
              `).join('')}
            </div>
          </div>
        </section>

        <section class="controls">
          <button class="tab-btn ${state.view === 'board' ? 'active' : ''}" data-view="board">Pre-Token Board (${filteredRows.length})</button>
          <button class="tab-btn ${state.view === 'live' ? 'active' : ''}" data-view="live">Live Token Anchors (${D.live.length})</button>
          <select id="tierSel" ${state.view !== 'board' ? 'disabled' : ''}>
            <option value="all" ${state.tier === 'all' ? 'selected' : ''}>All Pre-Token (${D.rows.length})</option>
            <option value="watchlist" ${state.tier === 'watchlist' ? 'selected' : ''}>Watchlist Only (${D.watchlist_size})</option>
            <option value="early" ${state.tier === 'early' ? 'selected' : ''}>Early &lt;$50M (${D.early_size})</option>
          </select>
          <select id="catSel" ${state.view !== 'board' ? 'disabled' : ''}>
            <option value="all">All Categories</option>
            ${categories.map((c) => `<option value="${esc(c)}" ${state.category === c ? 'selected' : ''}>${esc(c)}</option>`).join('')}
          </select>
          <select id="chainSel" ${state.view !== 'board' ? 'disabled' : ''}>
            <option value="all">All Chains</option>
            ${chains.map((c) => `<option value="${esc(c)}" ${state.chain === c ? 'selected' : ''}>${esc(c)}</option>`).join('')}
          </select>
          <input id="searchBox" type="search" placeholder="Search protocol name or slug…" value="${esc(state.search)}">
        </section>

        <section class="layout ${selected && state.view === 'board' ? 'with-detail' : ''}">
          <div class="table-wrap">
            ${state.view === 'board' ? `
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Protocol</th>
                    <th>Category</th>
                    <th>Chains</th>
                    <th class="num">TVL</th>
                    <th class="num">7d</th>
                    <th class="num">Cat Mult</th>
                    <th class="num">Implied Value</th>
                    <th class="num">Farm Plan</th>
                  </tr>
                </thead>
                <tbody>
                  ${filteredRows.slice(0, 80).map((r) => `
                    <tr class="data-row ${state.selectedSlug === r.slug ? 'selected' : ''}" data-slug="${esc(r.slug)}">
                      <td class="mono">${r.rank}</td>
                      <td>
                        <strong>${esc(r.name)}</strong>
                        ${r.on_watchlist ? '<span class="tag watch" style="margin-left:6px;">Watchlist</span>' : ''}
                      </td>
                      <td><span class="tag">${esc(r.category || 'Other')}</span></td>
                      <td style="color:var(--muted);font-size:12px;">${esc(r.chains.slice(0, 2).join(', '))}${r.chains.length > 2 ? ` +${r.chains.length - 2}` : ''}</td>
                      <td class="num">${fmtUsd(r.tvl_usd)}</td>
                      <td class="num">${fmtPct(r.change_7d)}</td>
                      <td class="num">${r.category_multiple.toFixed(2)}x</td>
                      <td class="num" style="color:#38bdf8;font-weight:700;">${fmtUsd(r.implied_value_usd)}</td>
                      <td class="num" style="font-size:12px;color:var(--muted);">$${r.farm_capital_usd} · ${r.farm_hold_days}d</td>
                    </tr>
                  `).join('')}
                </tbody>
              </table>
            ` : `
              <table>
                <thead>
                  <tr>
                    <th>Protocol</th>
                    <th>Symbol</th>
                    <th>Category</th>
                    <th class="num">TVL</th>
                    <th class="num">Market Cap</th>
                    <th class="num">MCap / TVL</th>
                    <th class="num">7d</th>
                  </tr>
                </thead>
                <tbody>
                  ${D.live.filter((l) => !state.search || l.name.toLowerCase().includes(state.search.toLowerCase())).slice(0, 80).map((l) => `
                    <tr>
                      <td><strong>${esc(l.name)}</strong></td>
                      <td class="mono">${esc(l.symbol || '—')}</td>
                      <td><span class="tag">${esc(l.category || 'Other')}</span></td>
                      <td class="num">${fmtUsd(l.tvl_usd)}</td>
                      <td class="num" style="color:#22c55e;font-weight:700;">${fmtUsd(l.mcap_usd)}</td>
                      <td class="num">${l.multiple !== null ? l.multiple.toFixed(2) + 'x' : '—'}</td>
                      <td class="num">${fmtPct(l.change_7d)}</td>
                    </tr>
                  `).join('')}
                </tbody>
              </table>
            `}
          </div>

          ${selected && state.view === 'board' ? `
            <aside class="detail">
              <div class="detail-head">
                <div>
                  <h3>#${selected.rank} ${esc(selected.name)}</h3>
                  <div class="subtitle">${esc(selected.category || 'Other')} · ${esc(selected.chains.join(', '))}</div>
                </div>
                <button class="close-btn" id="closeDetail">✕</button>
              </div>
              <div class="detail-metrics">
                <div class="dm"><div class="k">Implied Launch Value</div><div class="v" style="color:#38bdf8;">${fmtUsd(selected.implied_value_usd)}</div></div>
                <div class="dm"><div class="k">Current TVL</div><div class="v">${fmtUsd(selected.tvl_usd)}</div></div>
                <div class="dm"><div class="k">Category Multiple</div><div class="v">${selected.category_multiple.toFixed(3)}x</div></div>
                <div class="dm"><div class="k">Farm Budget / Hold</div><div class="v">$${selected.farm_capital_usd} / ${selected.farm_hold_days}d</div></div>
              </div>
              <div class="section-title">Costed Farm Plan</div>
              <ol class="steps">${buildFarmStepsHtml(selected)}</ol>
              <div class="section-title">Closest Live-Token Peer Anchors</div>
              <div class="peers">
                ${selected.peers.map((p) => `
                  <div class="peer">
                    <span><strong>${esc(p.name)}</strong> (${esc(p.symbol || '—')})</span>
                    <span class="mono">MCap ${fmtUsd(p.mcap_usd)} · ${p.multiple.toFixed(2)}x</span>
                  </div>
                `).join('')}
              </div>
            </aside>
          ` : ''}
        </section>
      </div>
    `;

    root.querySelectorAll('[data-view]').forEach((btn) => {
      btn.addEventListener('click', () => {
        state.view = btn.getAttribute('data-view');
        render();
      });
    });
    const tierSel = document.getElementById('tierSel');
    if (tierSel) tierSel.addEventListener('change', (e) => { state.tier = e.target.value; render(); });
    const catSel = document.getElementById('catSel');
    if (catSel) catSel.addEventListener('change', (e) => { state.category = e.target.value; render(); });
    const chainSel = document.getElementById('chainSel');
    if (chainSel) chainSel.addEventListener('change', (e) => { state.chain = e.target.value; render(); });
    const searchBox = document.getElementById('searchBox');
    if (searchBox) searchBox.addEventListener('input', (e) => { state.search = e.target.value; render(); });
    root.querySelectorAll('tr.data-row').forEach((tr) => {
      tr.addEventListener('click', () => {
        state.selectedSlug = tr.getAttribute('data-slug');
        render();
      });
    });
    const closeBtn = document.getElementById('closeDetail');
    if (closeBtn) closeBtn.addEventListener('click', () => { state.selectedSlug = null; render(); });
  }

  // Expose URL-hash hooks for deterministic screenshots (board, detail, filtered)
  if (location.hash.includes('detail=')) {
    const slug = new URLSearchParams(location.hash.slice(1)).get('detail');
    if (slug) state.selectedSlug = slug;
  }
  if (location.hash.includes('cat=')) {
    const cat = new URLSearchParams(location.hash.slice(1)).get('cat');
    if (cat) state.category = cat;
  }
  render();
})();
