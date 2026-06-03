"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  classifyAudio,
  getHealth,
  getMetrics,
  getMethods,
} from "@/lib/api";
import type {
  HealthResponse,
  MethodInfo,
  MethodResults,
  MetricRow,
} from "@/lib/types";
import Header from "./Header";
import SetupStatusBanner from "./SetupStatusBanner";
import MethodExplanation from "./MethodExplanation";
import MetricsTable from "./MetricsTable";
import AudioUploader from "./AudioUploader";
import ResultCards from "./ResultCards";
import MethodComparisonChart from "./MethodComparisonChart";
import PerMethodTabs from "./PerMethodTabs";
import { Card, CardContent } from "@/components/ui/card";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { InfoIcon, Loader2Icon, TriangleAlertIcon } from "lucide-react";

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-3 text-[11px] font-semibold tracking-[0.12em] text-muted-foreground uppercase">
      {children}
    </div>
  );
}

export default function ClassifierApp() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [methods, setMethods] = useState<MethodInfo[]>([]);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [bootError, setBootError] = useState<string | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [classifying, setClassifying] = useState(false);
  const [results, setResults] = useState<MethodResults | null>(null);
  const [classifyError, setClassifyError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);

  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const h = await getHealth();
        if (cancelled) return;
        setHealth(h);
        setBootError(null);
        getMetrics()
          .then((m) => !cancelled && setMetrics(m.rows))
          .catch(() => {});
        getMethods()
          .then((m) => !cancelled && setMethods(m.methods))
          .catch(() => {});
        if (!h.ready && h.required_ok && !h.load_error) {
          pollRef.current = setTimeout(poll, 3000);
        }
      } catch (e) {
        if (cancelled) return;
        setBootError(
          e instanceof Error ? e.message : "Cannot reach the backend."
        );
        pollRef.current = setTimeout(poll, 4000);
      }
    }

    poll();
    return () => {
      cancelled = true;
      if (pollRef.current) clearTimeout(pollRef.current);
    };
  }, []);

  const handleClassify = useCallback(async () => {
    if (!file) return;
    setClassifying(true);
    setClassifyError(null);
    setResults(null);
    setElapsed(null);
    const t0 = performance.now();
    try {
      const resp = await classifyAudio(file);
      setResults(resp.results);
      setElapsed((performance.now() - t0) / 1000);
      getHealth()
        .then((h) => setHealth(h))
        .catch(() => {});
      getMethods()
        .then((m) => setMethods(m.methods))
        .catch(() => {});
    } catch (e) {
      setClassifyError(
        e instanceof Error ? e.message : "Classification failed."
      );
    } finally {
      setClassifying(false);
    }
  }, [file]);

  const ready = !!health?.ready;
  const modelLoading = !!health && !health.ready && health.loading;
  const showSetupBanner =
    !!health && (!health.required_ok || !!health.load_error);
  const hasMlp = results ? "Supervised MLP" in results : true;

  return (
    <main className="mx-auto w-full max-w-5xl px-4 py-10 sm:px-6 sm:py-14">
      <Header />

      <div className="mt-10 flex flex-col gap-4">
        {bootError && !health && (
          <Alert variant="destructive">
            <TriangleAlertIcon />
            <AlertTitle>{bootError}</AlertTitle>
            <AlertDescription>
              Start it with{" "}
              <code className="text-foreground">
                uvicorn api:app --port 8000
              </code>
              . Retrying automatically…
            </AlertDescription>
          </Alert>
        )}

        {showSetupBanner && health && <SetupStatusBanner health={health} />}

        {modelLoading && (
          <Alert>
            <Loader2Icon className="animate-spin" />
            <AlertTitle>Loading the LAION-CLAP model…</AlertTitle>
            <AlertDescription>
              First start takes ~30 seconds while the model loads into memory.
            </AlertDescription>
          </Alert>
        )}

        {(methods.length > 0 || metrics.length > 0) && (
          <Card>
            <CardContent>
              <Accordion type="multiple">
                {methods.length > 0 && (
                  <AccordionItem value="methods">
                    <AccordionTrigger>How the methods work</AccordionTrigger>
                    <AccordionContent className="px-0.5 pt-2 pb-6">
                      <MethodExplanation methods={methods} />
                    </AccordionContent>
                  </AccordionItem>
                )}
                <AccordionItem value="metrics">
                  <AccordionTrigger>ESC-50 evaluation metrics</AccordionTrigger>
                  <AccordionContent className="px-0.5 pt-2 pb-6">
                    <MetricsTable rows={metrics} />
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </CardContent>
          </Card>
        )}
      </div>

      <section className="mt-12">
        <SectionLabel>Upload audio</SectionLabel>
        <AudioUploader
          file={file}
          onSelect={(f) => {
            setFile(f);
            setResults(null);
            setClassifyError(null);
          }}
          onClear={() => {
            setFile(null);
            setResults(null);
            setClassifyError(null);
          }}
          onClassify={handleClassify}
          busy={classifying}
          disabled={!ready}
        />
        {classifyError && (
          <Alert variant="destructive" className="mt-3">
            <TriangleAlertIcon />
            <AlertTitle>{classifyError}</AlertTitle>
          </Alert>
        )}
      </section>

      {results && (
        <section className="mt-12 flex flex-col gap-10">
          <div>
            <SectionLabel>Top prediction per method</SectionLabel>
            {!hasMlp && (
              <Alert className="mb-3">
                <InfoIcon />
                <AlertTitle>Supervised MLP is unavailable.</AlertTitle>
                <AlertDescription>
                  Run <code>python evaluate_methods.py</code> and restart the
                  backend to enable it.
                </AlertDescription>
              </Alert>
            )}
            <ResultCards results={results} />
            {elapsed !== null && (
              <p className="mt-3 text-center text-xs text-muted-foreground">
                Processed in {elapsed.toFixed(2)}s · CLAP audio embedding is
                computed once, then shared across methods.
              </p>
            )}
          </div>

          <div>
            <SectionLabel>Top-1 confidence comparison</SectionLabel>
            <MethodComparisonChart results={results} />
          </div>

          <div>
            <SectionLabel>Top-10 rankings by method</SectionLabel>
            <PerMethodTabs results={results} />
          </div>
        </section>
      )}

      <Separator className="mt-16" />
      <footer className="pt-6 pb-2 text-center text-xs text-muted-foreground">
        Based on “A Multimodal Prototypical Approach for Unsupervised Sound
        Classification” · INTERSPEECH 2023
      </footer>
    </main>
  );
}
