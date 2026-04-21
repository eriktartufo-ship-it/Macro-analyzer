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
