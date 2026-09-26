"use client";

import { useRef, useState } from "react";
import type { AnalysisResult, ApiError } from "./types";
import { RiskBadge } from "./risk-badge";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const OPTIC_DISC_METHOD_LABELS: Record<AnalysisResult["optic_disc_method"], string> = {
  TRAINED_MODEL: "Trained model",
  CV_BRIGHTEST_REGION: "Classical CV heuristic",
  FALLBACK_IMAGE_CENTER: "Fallback (image center, last resort)",
};

type Status = "idle" | "loading" | "success" | "error";

export default function Home() {
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    setResult(null);
    setError(null);
    setStatus("idle");
    if (file) {
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      setPreviewUrl(null);
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const file = fileInputRef.current?.files?.[0];
    if (!file) {
      setError("Choose a fundus image first.");
      return;
    }

    setStatus("loading");
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const body: ApiError = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(body.detail || `Request failed (${response.status})`);
      }

      const data: AnalysisResult = await response.json();
      setResult(data);
      setStatus("success");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not reach the API. Is it running at " + API_URL + "?"
      );
      setStatus("error");
    }
  }

  return (
    <div className="flex-1 bg-gray-50">
      <div className="mx-auto max-w-3xl px-6 py-12">
        <header className="mb-8">
          <h1 className="text-2xl font-bold text-gray-900">Retinal AVR Analysis</h1>
          <p className="mt-2 text-sm text-gray-600">
            Upload a fundus photograph to run the full pipeline: vessel segmentation, A/V
            classification, optic disc detection, and the scientific AVR (arteriolar-to-venular
            ratio) calculation used to estimate cardiovascular risk.
          </p>
        </header>

        <form
          onSubmit={handleSubmit}
          className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm"
        >
          <label
            htmlFor="image"
            className="mb-2 block text-sm font-medium text-gray-700"
          >
            Fundus image (JPG, PNG, or TIFF)
          </label>
          <input
            ref={fileInputRef}
            id="image"
            name="image"
            type="file"
            accept=".jpg,.jpeg,.png,.tif,.tiff,.bmp"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-700 file:mr-4 file:rounded-md file:border-0 file:bg-blue-50 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-blue-700 hover:file:bg-blue-100"
          />

          {previewUrl && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={previewUrl}
              alt="Fundus preview"
              className="mt-4 max-h-64 rounded-md border border-gray-200 object-contain"
            />
          )}

          <button
            type="submit"
            disabled={status === "loading"}
            className="mt-4 w-full rounded-md bg-blue-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {status === "loading" ? "Analyzing..." : "Analyze"}
          </button>
        </form>

        {error && (
          <div className="mt-6 rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-800">
            <strong className="font-semibold">Error:</strong> {error}
          </div>
        )}

        {result && (
          <div className="mt-6 space-y-6">
            <section className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900">AVR Result</h2>
                <RiskBadge level={result.risk_level} />
              </div>
              <div className="mt-4 flex items-baseline gap-2">
                <span className="text-4xl font-bold text-gray-900">
                  {result.avr.toFixed(3)}
                </span>
                <span className="text-sm text-gray-500">
                  AVR · confidence {result.avr_confidence.toLowerCase()}
                </span>
              </div>
              <p className="mt-2 text-sm text-gray-600">{result.risk_description}</p>

              <dl className="mt-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Stat label="CRAE" value={result.crae.toFixed(1)} />
                <Stat label="CRVE" value={result.crve.toFixed(1)} />
                <Stat label="Vessel area" value={`${result.vessel_percentage.toFixed(1)}%`} />
                <Stat
                  label="A/V confidence"
                  value={`${(result.confidence * 100).toFixed(1)}%`}
                />
              </dl>
            </section>

            <section className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
              <h2 className="text-lg font-semibold text-gray-900">Optic disc detection</h2>
              <dl className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Stat
                  label="Method"
                  value={OPTIC_DISC_METHOD_LABELS[result.optic_disc_method]}
                />
                <Stat
                  label="Confidence"
                  value={`${(result.optic_disc_confidence * 100).toFixed(0)}%`}
                />
                <Stat
                  label="Center (x, y)"
                  value={`${result.optic_disc_center[0].toFixed(0)}, ${result.optic_disc_center[1].toFixed(0)}`}
                />
                <Stat label="Radius" value={`${result.optic_disc_radius.toFixed(0)}px`} />
              </dl>
              {result.optic_disc_method === "FALLBACK_IMAGE_CENTER" && (
                <p className="mt-3 text-xs text-amber-700">
                  Warning: neither the trained model nor the CV heuristic were confident about
                  this image, so the peripapillary Zone B fell back to the image center, which is
                  not clinically valid. Treat this AVR value with caution.
                </p>
              )}
            </section>

            <section className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
              <h2 className="text-lg font-semibold text-gray-900">Performance</h2>
              <dl className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Stat
                  label="Segmentation"
                  value={`${result.inference_time_ms.toFixed(0)}ms`}
                />
                <Stat
                  label="A/V classification"
                  value={`${result.av_inference_time_ms.toFixed(0)}ms`}
                />
              </dl>
            </section>
          </div>
        )}

        <footer className="mt-12 text-center text-xs text-gray-400">
          Talks to the API at{" "}
          <code className="rounded bg-gray-100 px-1 py-0.5">{API_URL}</code>. See{" "}
          <code className="rounded bg-gray-100 px-1 py-0.5">docs/METRICS.md</code> in the
          repository for the model evaluation behind these numbers.
        </footer>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-wide text-gray-500">{label}</dt>
      <dd className="mt-1 text-sm font-semibold text-gray-900">{value}</dd>
    </div>
  );
}
