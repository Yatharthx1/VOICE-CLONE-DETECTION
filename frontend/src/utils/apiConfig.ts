/**
 * Centralized API & WebSocket configuration for VeraVoice.
 * Detects whether frontend is connecting via Vite dev server proxy,
 * direct port 8000, or served statically by FastAPI in production.
 */

export function getApiBaseUrl(): string {
  const envUrl = import.meta.env.VITE_API_URL;
  if (envUrl) {
    return envUrl.replace(/\/$/, '');
  }

  // If running in development (typically on port 5173), target port 8000 directly or via proxy
  if (typeof window !== 'undefined') {
    if (window.location.port === '5173') {
      return 'http://127.0.0.1:8000';
    }
    return window.location.origin;
  }
  return 'http://127.0.0.1:8000';
}

export function getWsBaseUrl(): string {
  const envWs = import.meta.env.VITE_WS_URL;
  if (envWs) {
    return envWs.replace(/\/$/, '');
  }

  const base = getApiBaseUrl();
  if (base.startsWith('https://')) {
    return base.replace('https://', 'wss://');
  }
  if (base.startsWith('http://')) {
    return base.replace('http://', 'ws://');
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}`;
}

export interface BackendHealth {
  status: string;
  version: string;
  framework: string;
  gpu_available: boolean;
  enrolled_speakers_count: number;
}

export interface BenchmarkSample {
  id: string;
  name: string;
  filename: string;
  tag: string;
  is_ai: boolean;
  description: string;
  duration_sec?: number;
  sample_rate?: number;
}

export async function checkBackendHealth(): Promise<{ ok: boolean; data?: BackendHealth; error?: string }> {
  try {
    const url = `${getApiBaseUrl()}/api/v1/health`;
    const res = await fetch(url, { method: 'GET' });
    if (!res.ok) {
      return { ok: false, error: `HTTP ${res.status}` };
    }
    const data: BackendHealth = await res.json();
    return { ok: true, data };
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : 'Backend unreachable';
    return { ok: false, error: msg };
  }
}

export async function fetchBenchmarkSamples(): Promise<BenchmarkSample[]> {
  try {
    const url = `${getApiBaseUrl()}/api/v1/samples`;
    const res = await fetch(url);
    if (!res.ok) {
      return [];
    }
    return await res.json();
  } catch {
    return [];
  }
}

export function getSampleAudioUrl(filename: string): string {
  return `${getApiBaseUrl()}/api/v1/samples/${encodeURIComponent(filename)}`;
}
