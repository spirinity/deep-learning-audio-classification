"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import {
  classifyAudio,
  getConfusion,
  getHealth,
  getMetrics,
  getMethods,
} from "@/lib/api";
import { UPLOAD_LIMITS } from "@/lib/constants";
import { COPY, type Language } from "@/lib/i18n";
import type {
  ConfusionResponse,
  HealthResponse,
  MethodInfo,
  MethodResults,
  MetricRow,
} from "@/lib/types";
import Header from "./Header";
import SetupStatusBanner from "./SetupStatusBanner";
import MethodExplanation from "./MethodExplanation";
import MetricsTable from "./MetricsTable";
import ConfusionMatrix from "./ConfusionMatrix";
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

const LANGUAGE_CHANGE_EVENT = "classifier-language-change";

function getClientLanguage(): Language {
  const saved = window.localStorage.getItem("language");
  if (saved === "en" || saved === "id") return saved;

  return window.navigator.language.toLowerCase().startsWith("id") ? "id" : "en";
}

function getServerLanguage(): Language {
  return "id";
}

function subscribeLanguage(onStoreChange: () => void) {
  window.addEventListener("storage", onStoreChange);
  window.addEventListener(LANGUAGE_CHANGE_EVENT, onStoreChange);

  return () => {
    window.removeEventListener("storage", onStoreChange);
    window.removeEventListener(LANGUAGE_CHANGE_EVENT, onStoreChange);
  };
}

function setStoredLanguage(language: Language) {
  window.localStorage.setItem("language", language);
  window.dispatchEvent(new Event(LANGUAGE_CHANGE_EVENT));
}

export default function ClassifierApp() {
  const language = useSyncExternalStore(
    subscribeLanguage,
    getClientLanguage,
    getServerLanguage
  );
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [methods, setMethods] = useState<MethodInfo[]>([]);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [confusion, setConfusion] = useState<ConfusionResponse | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [classifying, setClassifying] = useState(false);
  const [results, setResults] = useState<MethodResults | null>(null);
  const [classifyError, setClassifyError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);

  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const copy = COPY[language];
  const handleLanguageChange = useCallback((nextLanguage: Language) => {
    setStoredLanguage(nextLanguage);
  }, []);

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
        getConfusion()
          .then((c) => !cancelled && setConfusion(c))
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
      <Header
        copy={copy}
        language={language}
        onLanguageChange={handleLanguageChange}
      />

      <section className="mt-10 rounded-2xl border border-primary/25 bg-card px-4 py-5 ring-4 ring-primary/10 sm:px-6 sm:py-6">
        <div className="mb-4 flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <SectionLabel>{copy.uploadLabel}</SectionLabel>
            <h2 className="text-xl font-medium text-foreground">
              {copy.uploadTitle}
            </h2>
          </div>
          <p className="text-sm text-muted-foreground">
            {copy.uploadMeta} {UPLOAD_LIMITS.maxFileSizeMb} MB
          </p>
        </div>
        <AudioUploader
          copy={copy}
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

      <div className="mt-6 flex flex-col gap-4">
        {bootError && !health && (
          <Alert variant="destructive">
            <TriangleAlertIcon />
            <AlertTitle>{bootError}</AlertTitle>
            <AlertDescription>
              Start it with{" "}
              <code className="text-foreground">
                uvicorn api:app --port 8000
              </code>
              . {copy.backendRetry}
            </AlertDescription>
          </Alert>
        )}

        {showSetupBanner && health && (
          <SetupStatusBanner health={health} copy={copy} />
        )}

        {modelLoading && (
          <Alert>
            <Loader2Icon className="animate-spin" />
            <AlertTitle>{copy.modelLoadingTitle}</AlertTitle>
            <AlertDescription>{copy.modelLoadingDescription}</AlertDescription>
          </Alert>
        )}

        {(methods.length > 0 ||
          metrics.length > 0 ||
          !!confusion?.available) && (
          <Card>
            <CardContent>
              <Accordion type="multiple">
                {methods.length > 0 && (
                  <AccordionItem value="methods">
                    <AccordionTrigger>{copy.methodsWork}</AccordionTrigger>
                    <AccordionContent className="px-0.5 pt-2 pb-6">
                      <MethodExplanation methods={methods} copy={copy} />
                    </AccordionContent>
                  </AccordionItem>
                )}
                <AccordionItem value="metrics">
                  <AccordionTrigger>{copy.metricsTitle}</AccordionTrigger>
                  <AccordionContent className="px-0.5 pt-2 pb-6">
                    <MetricsTable rows={metrics} copy={copy} />
                  </AccordionContent>
                </AccordionItem>
                {confusion?.available && (
                  <AccordionItem value="confusion">
                    <AccordionTrigger>{copy.confusionTitle}</AccordionTrigger>
                    <AccordionContent className="px-0.5 pt-2 pb-6">
                      <ConfusionMatrix data={confusion} copy={copy} />
                    </AccordionContent>
                  </AccordionItem>
                )}
              </Accordion>
            </CardContent>
          </Card>
        )}
      </div>

      {results && (
        <section className="mt-12 flex flex-col gap-10">
          <div>
            <SectionLabel>{copy.topPrediction}</SectionLabel>
            {!hasMlp && (
              <Alert className="mb-3">
                <InfoIcon />
                <AlertTitle>{copy.mlpUnavailable}</AlertTitle>
                <AlertDescription>{copy.mlpEnable}</AlertDescription>
              </Alert>
            )}
            <ResultCards results={results} />
            {elapsed !== null && (
              <p className="mt-3 text-center text-xs text-muted-foreground">
                {copy.processedIn} {elapsed.toFixed(2)}s -{" "}
                {copy.processedSuffix}
              </p>
            )}
          </div>

          <div>
            <SectionLabel>{copy.confidenceComparison}</SectionLabel>
            <MethodComparisonChart results={results} />
          </div>

          <div>
            <SectionLabel>{copy.topRankings}</SectionLabel>
            <PerMethodTabs results={results} copy={copy} />
          </div>
        </section>
      )}

      <Separator className="mt-16" />
      <footer className="pt-6 pb-2 text-center text-xs text-muted-foreground">
        {copy.footer}
      </footer>
    </main>
  );
}
