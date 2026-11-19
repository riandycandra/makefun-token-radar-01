// Empirical, cross-validated token launch radar model for token-radar-01.
// Fits log10(mcap_usd) on live-token protocols in the snapshot using ridge-regularized OLS
// plus empirical-Bayes category shrinkage, validates out-of-sample with 5-fold CV, and scores
// every pre-token protocol with deposit-sized, costed farm plans.

export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

export interface ModelConstants {
  base_multiple: number;
  growth_weight: number;
  growth_floor: number;
  growth_cap: number;
  maturity_floor: number;
  maturity_range: number;
  maturity_days: number;
  unknown_age_days: number;
  shrinkage_k: number;
  ridge_lambda: number;
}

export const MODEL: ModelConstants = {
  base_multiple: 0.223,
  growth_weight: 0.35,
  growth_floor: -60,
  growth_cap: 120,
  maturity_floor: 0.75,
  maturity_range: 0.25,
  maturity_days: 1095,
  unknown_age_days: 365,
  shrinkage_k: 3,
  ridge_lambda: 6.0,
};

export const MODEL_VERSION = 'empirical-cv-2';
export const EARLY_ROWS = 260;
export const LIVE_ROWS = 128;

export interface FarmStep {
  text: string;
  cost: string;
  time: string;
}

export interface PeerAnchor {
  slug: string;
  name: string;
  symbol: string | null;
  tvl_usd: number;
  mcap_usd: number;
  multiple: number;
}

export interface FittedModelStats {
  live_sample_n: number;
  folds: number;
  cv_r2: number;
  cv_spearman: number;
  cv_mdape_pct: number;
  baseline_mdape_pct: number;
  global_median_multiple: number;
  weights: {
    intercept: number;
    log_tvl: number;
    log_chains: number;
    maturity: number;
    growth: number;
  };
  category_multiples: Array<{ category: string; n: number; multiple: number }>;
}

export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: 'pre-token';
  age_days: number | null;
  implied_value_usd: number;
  model: {
    base_multiple: number;
    category_multiple: number;
    growth_pct: number;
    growth_multiplier: number;
    maturity_multiplier: number;
    age_days: number | null;
  };
  farm_steps: string[];
}

export interface RadarRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  age_days: number | null;
  implied_value_usd: number;
  category_multiple: number;
  growth_multiplier: number;
  maturity_multiplier: number;
  on_watchlist: boolean;
  farm_capital_usd: number;
  farm_gas_usd: number;
  farm_hold_days: number;
  peers: PeerAnchor[];
}

export interface LiveRow {
  slug: string;
  name: string;
  symbol: string | null;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
  multiple: number | null;
}

export interface UniverseCounts {
  total: number;
  watchlist: number;
  pre_token_total: number;
  board_scored: number;
  live_with_mcap: number;
  token_without_mcap: number;
}

export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model: ModelConstants;
  model_version: string;
  watchlist_size: number;
  board_size: number;
  early_size: number;
  live_size: number;
  universe: UniverseCounts;
  fit: FittedModelStats;
  rows: RadarRow[];
  live: LiveRow[];
}

export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

const DAY_MS = 86_400_000;
const round4 = (n: number): number => Math.round(n * 1e4) / 1e4;

function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;

export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

export function isRankable(p: SnapshotProtocol): boolean {
  return isPreToken(p) && p.category !== 'CEX' && !VERSIONED.test(p.name);
}

/** Solve k x k linear system Ax = b via Gaussian elimination with partial pivoting. */
function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]!]);
  for (let i = 0; i < n; i++) {
    let maxR = i;
    for (let r = i + 1; r < n; r++) {
      if (Math.abs(M[r]![i]!) > Math.abs(M[maxR]![i]!)) maxR = r;
    }
    const tmp = M[i]!;
    M[i] = M[maxR]!;
    M[maxR] = tmp;
    const piv = M[i]![i]!;
    if (Math.abs(piv) < 1e-12) continue;
    for (let j = i; j <= n; j++) M[i]![j]! /= piv;
    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const f = M[r]![i]!;
      for (let j = i; j <= n; j++) M[r]![j]! -= f * M[i]![j]!;
    }
  }
  return M.map((row) => row[n]!);
}

function featuresOf(p: SnapshotProtocol, asOf: string, model: ModelConstants): number[] {
  const logTvl = Math.log10(Math.max(1, p.tvl_usd));
  const logChains = Math.log10(Math.max(1, p.chains.length) + 1);
  const age = ageDays(asOf, p.listed_at) ?? model.unknown_age_days;
  const mat = Math.min(Math.max(age, 0), model.maturity_days) / model.maturity_days;
  return [1, logTvl, logChains, mat, 0];
}

interface FitCore {
  w: number[];
  catAdj: Map<string, number>;
  catMult: Map<string, number>;
  globalMedMult: number;
}

function fitCore(train: SnapshotProtocol[], asOf: string, model: ModelConstants): FitCore {
  const d = 5;
  const XtX = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  const Xty = new Array<number>(d).fill(0);
  const ratios: number[] = [];

  for (const p of train) {
    const x = featuresOf(p, asOf, model);
    const y = Math.log10(Math.max(1, p.mcap_usd!));
    ratios.push(p.mcap_usd! / p.tvl_usd);
    for (let i = 0; i < d; i++) {
      Xty[i]! += x[i]! * y;
      for (let j = 0; j < d; j++) XtX[i]![j]! += x[i]! * x[j]!;
    }
  }
  for (let i = 1; i < d; i++) XtX[i]![i]! += model.ridge_lambda;
  const w = solveLinear(XtX, Xty);

  // Compute residuals per category and apply empirical Bayes shrinkage
  const resByCat = new Map<string, number[]>();
  const ratioByCat = new Map<string, number[]>();
  for (const p of train) {
    const x = featuresOf(p, asOf, model);
    const y = Math.log10(Math.max(1, p.mcap_usd!));
    const pred = x.reduce((acc, xi, idx) => acc + xi * w[idx]!, 0);
    const cat = p.category ?? 'Other';
    const arr = resByCat.get(cat) ?? [];
    arr.push(y - pred);
    resByCat.set(cat, arr);

    const rArr = ratioByCat.get(cat) ?? [];
    rArr.push(Math.log10(p.mcap_usd! / p.tvl_usd));
    ratioByCat.set(cat, rArr);
  }

  ratios.sort((a, b) => a - b);
  const globalMedMult = ratios.length > 0 ? ratios[Math.floor(ratios.length / 2)]! : model.base_multiple;
  const globalLogMed = Math.log10(Math.max(0.01, globalMedMult));

  const catAdj = new Map<string, number>();
  const catMult = new Map<string, number>();
  for (const [cat, resList] of resByCat.entries()) {
    const meanRes = resList.reduce((a, b) => a + b, 0) / resList.length;
    const shrunkRes = (resList.length / (resList.length + model.shrinkage_k)) * meanRes;
    catAdj.set(cat, shrunkRes);

    const rList = (ratioByCat.get(cat) ?? []).sort((a, b) => a - b);
    const medLog = rList.length > 0 ? rList[Math.floor(rList.length / 2)]! : globalLogMed;
    const shrunkLog = (rList.length * medLog + model.shrinkage_k * globalLogMed) / (rList.length + model.shrinkage_k);
    catMult.set(cat, Math.pow(10, shrunkLog));
  }

  return { w, catAdj, catMult, globalMedMult };
}

function predictWithFit(p: SnapshotProtocol, asOf: string, model: ModelConstants, core: FitCore): number {
  const x = featuresOf(p, asOf, model);
  const baseLog = x.reduce((acc, xi, idx) => acc + xi * core.w[idx]!, 0);
  const adj = core.catAdj.get(p.category ?? 'Other') ?? 0;
  const catM = core.catMult.get(p.category ?? 'Other') ?? core.globalMedMult;
  const age = ageDays(asOf, p.listed_at) ?? model.unknown_age_days;
  const matMult = model.maturity_floor + (Math.min(Math.max(age, 0), model.maturity_days) / model.maturity_days) * model.maturity_range;
  const gPct = Math.min(model.growth_cap, Math.max(model.growth_floor, finite(p.change_7d, 0)));
  const gMult = 1 + (gPct / 100) * model.growth_weight;
  const multLog = Math.log10(Math.max(1, p.tvl_usd * catM * matMult * gMult));
  return Math.max(1, Math.round(Math.pow(10, 0.55 * (baseLog + adj) + 0.45 * multLog)));
}

function spearman(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n < 3) return 0;
  const ranks = (arr: number[]) => {
    const idx = arr.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
    const r = new Array<number>(n);
    for (let i = 0; i < n; i++) r[idx[i]!.i] = i + 1;
    return r;
  };
  const rx = ranks(xs);
  const ry = ranks(ys);
  let d2 = 0;
  for (let i = 0; i < n; i++) d2 += (rx[i]! - ry[i]!) ** 2;
  return 1 - (6 * d2) / (n * (n * n - 1));
}

function fitAndValidate(snapshot: Snapshot, model: ModelConstants): { core: FitCore; stats: FittedModelStats } {
  const live = snapshot.protocols
    .filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0)
    .sort((a, b) => (a.slug < b.slug ? -1 : 1));

  const folds = 5;
  const yTrueLog: number[] = [];
  const yPredLog: number[] = [];
  const apeModel: number[] = [];
  const apeBase: number[] = [];

  for (let f = 0; f < folds; f++) {
    const train = live.filter((_, i) => i % folds !== f);
    const test = live.filter((_, i) => i % folds === f);
    const foldCore = fitCore(train, snapshot.as_of, model);
    for (const p of test) {
      const pred = predictWithFit(p, snapshot.as_of, model, foldCore);
      const actual = p.mcap_usd!;
      yTrueLog.push(Math.log10(actual));
      yPredLog.push(Math.log10(pred));
      apeModel.push(Math.abs(pred - actual) / actual);
      const basePred = p.tvl_usd * 0.25;
      apeBase.push(Math.abs(basePred - actual) / actual);
    }
  }

  const meanY = yTrueLog.reduce((a, b) => a + b, 0) / Math.max(1, yTrueLog.length);
  const ssTot = yTrueLog.reduce((a, y) => a + (y - meanY) ** 2, 0);
  const ssRes = yTrueLog.reduce((a, y, i) => a + (y - yPredLog[i]!) ** 2, 0);
  const cvR2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;

  apeModel.sort((a, b) => a - b);
  apeBase.sort((a, b) => a - b);
  const medModel = apeModel.length > 0 ? apeModel[Math.floor(apeModel.length / 2)]! * 100 : 0;
  const medBase = apeBase.length > 0 ? apeBase[Math.floor(apeBase.length / 2)]! * 100 : 0;

  const fullCore = fitCore(live, snapshot.as_of, model);
  const catList = [...fullCore.catMult.entries()]
    .map(([category, multiple]) => ({
      category,
      n: live.filter((p) => (p.category ?? 'Other') === category).length,
      multiple: round4(multiple),
    }))
    .sort((a, b) => b.n - a.n)
    .slice(0, 10);

  return {
    core: fullCore,
    stats: {
      live_sample_n: live.length,
      folds,
      cv_r2: round4(cvR2),
      cv_spearman: round4(spearman(yTrueLog, yPredLog)),
      cv_mdape_pct: Math.round(medModel * 10) / 10,
      baseline_mdape_pct: Math.round(medBase * 10) / 10,
      global_median_multiple: round4(fullCore.globalMedMult),
      weights: {
        intercept: round4(fullCore.w[0]!),
        log_tvl: round4(fullCore.w[1]!),
        log_chains: round4(fullCore.w[2]!),
        maturity: round4(fullCore.w[3]!),
        growth: round4(fullCore.w[4]!),
      },
      category_multiples: catList,
    },
  };
}

export function stepText(step: FarmStep): string {
  let s = `${step.text} Cost: ${step.cost}. Time: ${step.time}.`;
  if (s.length > 600) s = `${s.slice(0, 596)}...`;
  if (s.length < 20) s = `${s} Hold the position and check the protocol's own page.`;
  return s;
}

function chainOf(p: SnapshotProtocol): string {
  return p.chains.length > 0 ? p.chains[0]! : 'its main chain';
}

function capitalFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 5_000_000_000) return 500;
  if (p.tvl_usd >= 1_000_000_000) return 250;
  if (p.tvl_usd >= 100_000_000) return 150;
  return 75;
}

function gasFor(chain: string): number {
  const c = chain.toLowerCase();
  if (c === 'ethereum') return 12;
  if (c === 'solana' || c === 'sui' || c === 'aptos' || c === 'base' || c === 'arbitrum' || c === 'hyperliquid l1') return 1;
  return 3;
}

function holdDaysFor(age: number | null): number {
  if (age === null) return 30;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

function actionFor(p: SnapshotProtocol, capital: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) return `Supply $${capital} of USDC to ${name} on ${chain} and keep the position open.`;
  if (c.includes('dex') || c.includes('yield') || c.includes('leveraged farm')) return `Provide $${capital} of liquidity to ${name}'s core pool on ${chain} and keep the LP receipt.`;
  if (c.includes('bridge')) return `Bridge $${capital} of USDC to ${chain} via ${name} and execute one follow-up swap.`;
  if (c.includes('staking') || c.includes('restak')) return `Stake $${capital} of the native asset via ${name} on ${chain} and hold the receipt token.`;
  if (c.includes('perpetual') || c.includes('derivative')) return `Deposit $${capital} of USDC margin on ${name} (${chain}) and maintain weekly open interest.`;
  if (c.includes('prediction')) return `Deposit $${capital} on ${name} and place balanced market positions across active events.`;
  if (c.includes('curator') || c.includes('allocator')) return `Deposit $${capital} into a curated ${name} vault on ${chain} and hold the vault shares.`;
  if (c.includes('stable') || c.includes('basis')) return `Mint or deposit $${capital} of the protocol's stable asset on ${chain} via ${name}.`;
  if (c.includes('rwa')) return `Mint $${capital} of ${name}'s yield-bearing RWA token on ${chain} and hold through snapshots.`;
  return `Deposit $${capital} of USDC into ${name} on ${chain} and keep the position active.`;
}

export function farmSteps(p: SnapshotProtocol, age: number | null): FarmStep[] {
  const chain = chainOf(p);
  const capital = capitalFor(p);
  const gas = gasFor(chain);
  const hold = holdDaysFor(age);
  const action = actionFor(p, capital, chain);
  return [
    {
      text: `Isolate a dedicated farming wallet funded on ${chain} with $${capital} USDC plus gas reserve.`,
      cost: `$${gas} initial gas on ${chain}`,
      time: '5 minutes',
    },
    {
      text: action,
      cost: `$${capital} deposit + ~$${gas} gas`,
      time: '15 minutes setup',
    },
    {
      text: `Maintain steady weekly cadence: complete one qualifying interaction (borrow, rebalance, swap, or vote) per week.`,
      cost: `~$${Math.max(1, Math.round(gas / 2))} gas/week`,
      time: '10 minutes/week',
    },
    {
      text: `Hold the $${capital} position across at least ${hold} days to span multiple snapshot epochs.`,
      cost: `$${capital} capital locked (${hold}d)`,
      time: `${hold} days`,
    },
    {
      text: `Verify eligibility only on ${p.name}'s official points or governance portal before unwinding.`,
      cost: 'no fee to verify',
      time: '10 minutes',
    },
    {
      text: `Record entry tx, wallet address, and snapshot dates in your farm ledger for tax and sybil hygiene.`,
      cost: '$0',
      time: '5 minutes',
    },
  ];
}

export function liveRows(snapshot: Snapshot): LiveRow[] {
  return snapshot.protocols
    .filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.category !== 'CEX')
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd)
    .slice(0, LIVE_ROWS)
    .map((p) => ({
      slug: p.slug,
      name: p.name,
      symbol: p.symbol,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      mcap_usd: p.mcap_usd,
      change_7d: p.change_7d,
      multiple: p.tvl_usd > 0 && p.mcap_usd ? round4(p.mcap_usd / p.tvl_usd) : null,
    }));
}

function peersFor(p: SnapshotProtocol, liveAll: SnapshotProtocol[]): PeerAnchor[] {
  const cat = p.category ?? '';
  let pool = liveAll.filter((l) => l.category === cat && l.mcap_usd && l.tvl_usd > 0);
  if (pool.length < 2) pool = liveAll.filter((l) => l.mcap_usd && l.tvl_usd > 0);
  return [...pool]
    .sort((a, b) => Math.abs(Math.log10(a.tvl_usd) - Math.log10(p.tvl_usd)) - Math.abs(Math.log10(b.tvl_usd) - Math.log10(p.tvl_usd)))
    .slice(0, 3)
    .map((l) => ({
      slug: l.slug,
      name: l.name,
      symbol: l.symbol,
      tvl_usd: l.tvl_usd,
      mcap_usd: l.mcap_usd!,
      multiple: round4(l.mcap_usd! / l.tvl_usd),
    }));
}

export function buildBoard(snapshot: Snapshot, model: ModelConstants = MODEL): { board: BoardRow[]; stats: FittedModelStats; core: FitCore } {
  const { core, stats } = fitAndValidate(snapshot, model);
  const index = new Map(snapshot.protocols.map((p) => [p.slug, p]));
  const chosen = new Map<string, SnapshotProtocol>();
  for (const slug of snapshot.watchlist) {
    const p = index.get(slug);
    if (p !== undefined) chosen.set(slug, p);
  }
  const early = snapshot.protocols
    .filter((p) => !chosen.has(p.slug) && isRankable(p) && p.tvl_usd < snapshot.floor_usd)
    .sort((a, b) => b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0))
    .slice(0, EARLY_ROWS);
  for (const p of early) chosen.set(p.slug, p);

  const scored = [...chosen.values()].map((p) => {
    const implied = predictWithFit(p, snapshot.as_of, model, core);
    const age = ageDays(snapshot.as_of, p.listed_at);
    const growthPct = Math.min(model.growth_cap, Math.max(model.growth_floor, finite(p.change_7d, 0)));
    const growthMult = 1 + (growthPct / 100) * model.growth_weight;
    const ageForModel = age ?? model.unknown_age_days;
    const maturityMult = model.maturity_floor + (Math.min(Math.max(ageForModel, 0), model.maturity_days) / model.maturity_days) * model.maturity_range;
    const catMult = core.catMult.get(p.category ?? 'Other') ?? core.globalMedMult;
    return { p, implied, growthPct, growthMult, maturityMult, age, catMult };
  });

  scored.sort((a, b) => b.implied - a.implied || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));
  const board: BoardRow[] = scored.map((s, i) => ({
    rank: i + 1,
    slug: s.p.slug,
    name: s.p.name,
    category: s.p.category,
    chains: s.p.chains,
    tvl_usd: s.p.tvl_usd,
    change_7d: s.p.change_7d,
    mcap_usd: s.p.mcap_usd,
    listed_at: s.p.listed_at,
    gecko_id: s.p.gecko_id,
    symbol: s.p.symbol,
    token_status: 'pre-token',
    age_days: s.age,
    implied_value_usd: s.implied,
    model: {
      base_multiple: round4(core.globalMedMult),
      category_multiple: round4(s.catMult),
      growth_pct: s.growthPct,
      growth_multiplier: round4(s.growthMult),
      maturity_multiplier: round4(s.maturityMult),
      age_days: s.age,
    },
    farm_steps: farmSteps(s.p, s.age).map(stepText),
  }));

  return { board, stats, core };
}

export function buildRadar(snapshot: Snapshot, model: ModelConstants = MODEL): { board: BoardRow[]; radar: RadarData } {
  const { board, stats } = buildBoard(snapshot, model);
  const live = liveRows(snapshot);
  const liveAll = snapshot.protocols.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0);
  const watchSet = new Set(snapshot.watchlist);
  const early = board.filter((r) => !watchSet.has(r.slug)).length;

  const preTotal = snapshot.protocols.filter((p) => isPreToken(p)).length;
  const liveMcapCount = snapshot.protocols.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0).length;

  const rows: RadarRow[] = board.map((r) => {
    const cap = capitalFor(r);
    const ch = chainOf(r);
    return {
      rank: r.rank,
      slug: r.slug,
      name: r.name,
      category: r.category,
      chains: r.chains,
      tvl_usd: r.tvl_usd,
      change_7d: r.change_7d,
      mcap_usd: r.mcap_usd,
      listed_at: r.listed_at,
      age_days: r.age_days,
      implied_value_usd: r.implied_value_usd,
      category_multiple: r.model.category_multiple,
      growth_multiplier: r.model.growth_multiplier,
      maturity_multiplier: r.model.maturity_multiplier,
      on_watchlist: watchSet.has(r.slug),
      farm_capital_usd: cap,
      farm_gas_usd: gasFor(ch),
      farm_hold_days: holdDaysFor(r.age_days),
      peers: peersFor(r, liveAll),
    };
  });

  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model,
    model_version: MODEL_VERSION,
    watchlist_size: snapshot.watchlist.length,
    board_size: board.length,
    early_size: early,
    live_size: live.length,
    universe: {
      total: snapshot.protocols.length,
      watchlist: snapshot.watchlist.length,
      pre_token_total: preTotal,
      board_scored: board.length,
      live_with_mcap: liveMcapCount,
      token_without_mcap: snapshot.protocols.length - preTotal - liveMcapCount,
    },
    fit: stats,
    rows,
    live,
  };
  return { board, radar };
}

export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return {
    as_of: o.as_of,
    source: o.source,
    floor_usd: o.floor_usd,
    protocols,
    watchlist: o.watchlist as string[],
  };
}

export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slug${missing.length === 1 ? '' : 's'}: ${missing.slice(0, 5).join(', ')}`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all ${board.length} rows share the same implied value $${board[0]!.implied_value_usd}: a board that ranks nothing by value is not a ranking`;
  }
  for (const r of board) {
    const steps = r.farm_steps;
    if (steps.length < 3 || steps.length > 12) return `${r.slug} has ${steps.length} farm steps`;
    for (const s of steps) if (s.length < 20 || s.length > 600) return `${r.slug} has a farm step of ${s.length} characters`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
  }
  return null;
}

export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}
