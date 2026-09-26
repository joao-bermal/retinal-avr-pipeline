// Mirrors the JSON shape returned by POST /analyze in api/main.py.
// Keep in sync manually -- there is no shared schema between the Python
// backend and this frontend.
export interface AnalysisResult {
  image_path: string;
  vessel_percentage: number;
  inference_time_ms: number;
  confidence: number;
  av_inference_time_ms: number;
  optic_disc_center: [number, number];
  optic_disc_radius: number;
  optic_disc_method: "TRAINED_MODEL" | "CV_BRIGHTEST_REGION" | "FALLBACK_IMAGE_CENTER";
  optic_disc_confidence: number;
  avr: number;
  crae: number;
  crve: number;
  risk_level: "NORMAL" | "BORDERLINE" | "HIGH_RISK" | "VERY_HIGH_RISK" | "INDETERMINATE";
  risk_description: string;
  avr_confidence: "HIGH" | "MEDIUM" | "LOW";
  avr_status: "SUCCESS" | "INSUFFICIENT_VESSELS" | "ERROR";
  avr_method: string;
}

export interface ApiError {
  detail: string;
}
