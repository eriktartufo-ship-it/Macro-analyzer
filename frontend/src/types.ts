export type Regime =
  | "reflation"
  | "stagflation"
  | "deflation"
  | "goldilocks";

export interface RegimeProbabilities {
  reflation: number;
  stagflation: number;
  deflation: number;
  goldilocks: number;
}

export interface CurrentRegime {
  date: string;
  regime: Regime;
  probabilities: RegimeProbabilities;
  confidence: number;
  fit_scores?: Partial<RegimeProbabilities>;
}

export type RegimeHistoryItem = CurrentRegime;

export interface Scoreboard {
  date: string;
  regime: Regime;
  confidence: number;
  scores: Record<string, number>;
}

export interface PlayerSignal {
  key: string;
  label: string;
  value: number | null;
  score: number | null;
  interpret: string;
  red_flag: boolean;
  weight: number;
}

export interface PlayerScore {
  label: string;
  score: number;
  coverage: number;
  signals: PlayerSignal[];
}

export interface Dedollarization {
  date: string;
  score: number;
  structural_score: number;
  decade_score: number;
  twenty_year_score: number | null;
  acceleration: number;
  combined_score: number;
  geopolitical_score: number;
  components: Record<string, number>;
  structural: Record<string, number>;
  decade: Record<string, number>;
  twenty_year: Record<string, number>;
  by_player: Record<string, PlayerScore>;
  player_history: Record<string, Record<string, number>>;
  player_acceleration: Record<string, number>;
  explanation?: string | null;
}

export interface ConditionDetail {
  name: string;
  description: string;
  raw_score: number;
  weight: number;
  signal: "strong" | "moderate" | "weak" | "neutral";
}

export interface TrajectoryForce {
  name: string;
  type: "indicator" | "news" | "dedollarization";
  description: string;
  pushes_toward: string;
  strength: number;
}

export interface TrajectoryDrift {
  regime: string;
  current: number;
  projected: number;
  delta: number;
}

export interface Trajectory {
  current_regime: string;
  projected_regime: string;
  projected_probabilities: Record<string, number>;
  projected_fit_scores?: Record<string, number>;
  forces: TrajectoryForce[];
  drift: TrajectoryDrift[];
  transition_risk: number;
  summary: string;
  projected_scores: Record<string, number>;
}

export interface NewsItem {
  date: string;
  source: string;
  title: string;
  summary: string;
  sentiment: number;
  relevance: number;
  affected_assets: Record<string, number>;
}

export interface DedollarHistoryItem {
  date: string;
  combined_score: number;
  cyclical_score: number;
  structural_score: number;
  decade_score: number;
  twenty_year_score: number | null;
  geopolitical_score: number;
  acceleration: number;
}

export interface SignalsHistoryItem {
  date: string;
  scores: Record<string, number>;
}

export interface MacroIndicatorsHistoryItem {
  date: string;
  indicators: Record<string, number>;
}

export interface PlayerHistoryItem {
  date: string;
  players: Record<string, number>;
}

export interface RegimeConditionMeta {
  name: string;
  weight: number;
  description: string;
}

export interface AssetRegimeStats {
  hit_rate: number;
  avg_return: number;
  vol: number;
  sharpe: number;
}

export interface DataSnapshot {
  date: string | null;
  regime: Regime | null;
  regime_conditions: Record<string, RegimeConditionMeta[]>;
  asset_regime_data: Record<string, Record<string, AssetRegimeStats>>;
  asset_dedollar_sensitivity: Record<string, number>;
  current_indicators: Record<string, number>;
  current_dedollar_indicators: Record<string, number>;
  current_probabilities: Record<string, number>;
  current_fit_scores: Record<string, number>;
  current_asset_scores: Record<string, number>;
  current_dedollar_combined: number | null;
}

export interface TransitionMatrix {
  horizon_days: number;
  regimes: string[];
  counts: Record<string, Record<string, number>>;
  probabilities: Record<string, Record<string, number>>;
  avg_duration_days: Record<string, number>;
  self_transition_probability: Record<string, number>;
  total_observations: number;
  date_from: string | null;
  date_to: string | null;
  projected_probabilities: Record<string, number> | null;
}

export interface HMMPrediction {
  regimes: string[];
  probabilities: Record<string, number>;
  current_state: number;
  state_to_regime: Record<number, string>;
  n_training: number;
  log_likelihood: number;
  feature_means: Record<string, number>;
  feature_stds: Record<string, number>;
}

export interface TermPremiumPoint {
  date: string;
  fitted_yield: number;
  term_premium: number;
  expected_path: number;
}

export interface TermPremiumStat {
  regime: string;
  n_observations: number;
  mean_fitted: number | null;
  mean_term_premium: number | null;
  mean_expected_path: number | null;
  pct_term_premium_positive: number | null;
}

export interface TermPremiumReport {
  common_period: [string, string];
  threshold: number;
  n_observations: number;
  points: TermPremiumPoint[];
  by_regime: TermPremiumStat[];
}

export interface FactorRegimeStat {
  factor: string;
  regime: string;
  n_observations: number;
  mean_annual: number | null;
  vol_annual: number | null;
  sharpe: number | null;
  win_rate: number | null;
}

export interface FactorRegimeReport {
  threshold: number;
  n_months_analyzed: number;
  factor_keys: string[];
  regimes: string[];
  common_period: [string, string];
  stats: FactorRegimeStat[];
}

export interface SmoothedSeriesPoint {
  date: string;
  raw: number;
  filtered: number;
  smoothed: number;
}

export interface SmoothedIndicator {
  series_name: string;
  description: string;
  lambda_used: number;
  n_points: number;
  variance_reduction: number;
  points: SmoothedSeriesPoint[];
}

export interface SmoothableIndicator {
  key: string;
  description: string;
}

export interface ScoreComparisonItem {
  asset: string;
  pure_score: number;
  adjusted_score: number;
  dedollar_delta: number;
  sensitivity: number;
}

export interface DedollarComparison {
  date: string;
  regime: string;
  probabilities: Record<string, number>;
  dedollar_combined_score: number;
  use_dedollar_bonus_active: boolean;
  items: ScoreComparisonItem[];
}

export interface BandSeries {
  median: number[];
  p10: number[];
  p25: number[];
  p75: number[];
  p90: number[];
  mean: number[];
}

export interface RegimeBand extends BandSeries {
  regime: string;
}

export interface AssetBand extends BandSeries {
  asset: string;
}

export interface MonteCarloForecast {
  n_paths: number;
  n_steps: number;
  horizon_days: number;
  initial_distribution: Record<string, number>;
  step_dates_offsets: number[];
  transition_matrix_observations: number;
  regime_bands: RegimeBand[];
  asset_bands: AssetBand[];
  notes: string[];
}

export interface ScenarioPreset {
  key: string;
  label: string;
  description: string;
  deltas: Record<string, { op: string; value: number }>;
}

export interface ScenarioResult {
  scenario_key: string;
  label: string;
  description: string;
  baseline_indicators: Record<string, number>;
  shocked_indicators: Record<string, number>;
  baseline_regime: string;
  baseline_probabilities: Record<string, number>;
  shocked_regime: string;
  shocked_probabilities: Record<string, number>;
  baseline_scores: Record<string, number>;
  shocked_scores: Record<string, number>;
  asset_score_deltas: Record<string, number>;
}

export interface PerfStats {
  total_return: number;
  annualized_return: number;
  annualized_volatility: number;
  sharpe: number;
  max_drawdown: number;
  calmar: number;
  win_rate: number;
  n_months: number;
  start_date: string;
  end_date: string;
  final_nav: number;
}

export interface BacktestStrategy {
  name: string;
  description: string;
  nav: { date: string; value: number }[];
  monthly_returns: { date: string; value: number }[];
  stats: PerfStats;
  alpha_vs_60_40: { alpha: number; beta: number; correlation: number };
}

export interface BacktestResult {
  common_start: string;
  common_end: string;
  n_months: number;
  strategies: BacktestStrategy[];
}

export interface RecessionLead {
  recession_start: string;
  recession_end: string;
  duration_months: number;
  signal_date: string | null;
  lead_months: number | null;
  max_prob_during: number;
  pre_recession_max_prob: number;
}

export interface LeadTimeReport {
  threshold: number;
  lookback_months: number;
  n_recessions_analyzed: number;
  hit_rate: number;
  avg_lead_months: number | null;
  median_lead_months: number | null;
  recessions: RecessionLead[];
}

export interface EnsembleModelView {
  name: string;
  probabilities: Record<string, number>;
  error: string | null;
  metadata: Record<string, unknown> | null;
}

export interface EnsembleResult {
  weights: Record<string, number>;
  views: EnsembleModelView[];
  ensemble_probabilities: Record<string, number>;
  confidence: number;
  disagreement_score: number;
  high_disagreement: boolean;
  dominant_regime: string;
  notes: string[];
}

export interface CalibrationDiagnostic {
  asset: string;
  regime: string;
  n_observations: number;
  weight_measured: number;
  prior: { hit_rate: number; avg_return: number; vol: number; sharpe: number };
  measured: {
    hit_rate: number | null;
    real_return: number | null;
    vol: number | null;
    sharpe: number | null;
  } | null;
  calibrated: { hit_rate: number; avg_return: number; vol: number; sharpe: number };
}

export interface CalibrationPayload {
  version: number;
  calibrated_on: string;
  n_classifications: number;
  params: Record<string, number>;
  diagnostics: CalibrationDiagnostic[];
  use_calibrated_scoring: boolean;
}

export interface RegimeExplain {
  date: string;
  regime: Regime;
  probabilities: RegimeProbabilities;
  fit_scores?: Partial<RegimeProbabilities>;
  confidence: number;
  indicators: Record<string, number>;
  top_drivers: ConditionDetail[];
  dedollar_indicators: Record<string, number>;
  trajectory: Trajectory | null;
}
