// Thin client for the FastAPI backend (api.py).
// Base URL is configurable via NEXT_PUBLIC_API_BASE; defaults to local dev.

import type {
  ClassifyResponse,
  HealthResponse,
  MethodsResponse,
  MetricsResponse,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8000";

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { signal });
  if (!res.ok) {
    throw new ApiError(await extractDetail(res), res.status);
  }
  return res.json() as Promise<T>;
}

async function extractDetail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (body && typeof body.detail === "string") return body.detail;
  } catch {
    /* ignore non-JSON bodies */
  }
  return `Request failed (${res.status})`;
}

export function getHealth(signal?: AbortSignal) {
  return getJson<HealthResponse>("/api/health", signal);
}

export function getMethods(signal?: AbortSignal) {
  return getJson<MethodsResponse>("/api/methods", signal);
}

export function getMetrics(signal?: AbortSignal) {
  return getJson<MetricsResponse>("/api/metrics", signal);
}

export async function classifyAudio(file: File): Promise<ClassifyResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/classify`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    throw new ApiError(await extractDetail(res), res.status);
  }
  return res.json() as Promise<ClassifyResponse>;
}

export { ApiError };
