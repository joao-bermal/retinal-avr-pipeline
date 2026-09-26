import type { AnalysisResult } from "./types";

const RISK_STYLES: Record<AnalysisResult["risk_level"], string> = {
  NORMAL: "bg-emerald-100 text-emerald-800 border-emerald-300",
  BORDERLINE: "bg-amber-100 text-amber-800 border-amber-300",
  HIGH_RISK: "bg-orange-100 text-orange-800 border-orange-300",
  VERY_HIGH_RISK: "bg-red-100 text-red-800 border-red-300",
  INDETERMINATE: "bg-gray-100 text-gray-700 border-gray-300",
};

const RISK_LABELS: Record<AnalysisResult["risk_level"], string> = {
  NORMAL: "Normal",
  BORDERLINE: "Borderline",
  HIGH_RISK: "High risk",
  VERY_HIGH_RISK: "Very high risk",
  INDETERMINATE: "Indeterminate",
};

export function RiskBadge({ level }: { level: AnalysisResult["risk_level"] }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-3 py-1 text-sm font-semibold ${RISK_STYLES[level]}`}
    >
      {RISK_LABELS[level]}
    </span>
  );
}
