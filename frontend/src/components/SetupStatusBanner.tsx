import type { HealthResponse } from "@/lib/types";
import type { Copy } from "@/lib/i18n";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2Icon, TriangleAlertIcon, XCircleIcon } from "lucide-react";

interface Props {
  health: HealthResponse;
  copy: Copy;
}

// Only shown when required files are missing or the model failed to load.
export default function SetupStatusBanner({ health, copy }: Props) {
  const hasError = !!health.load_error;

  return (
    <Alert variant="destructive">
      <TriangleAlertIcon />
      <AlertTitle>
        {hasError
          ? copy.setupLoadFailed
          : copy.setupFilesMissing}
      </AlertTitle>
      <AlertDescription>
        {hasError && (
          <pre className="w-full overflow-x-auto rounded-md bg-background/60 p-3 text-xs">
            {health.load_error}
          </pre>
        )}

        <ul className="mt-1 flex w-full flex-col gap-1.5">
          {health.files.map((f) => (
            <li key={f.item} className="flex items-center justify-between gap-3">
              <span className="flex items-center gap-2 text-foreground">
                {f.exists ? (
                  <CheckCircle2Icon className="size-4 text-chart-3" />
                ) : (
                  <XCircleIcon className="size-4 text-destructive" />
                )}
                {f.item}
                {!f.required && (
                  <Badge variant="secondary" className="text-[10px]">
                    {copy.optional}
                  </Badge>
                )}
              </span>
              <code className="truncate text-xs text-muted-foreground">
                {f.path}
              </code>
            </li>
          ))}
        </ul>

        <p className="mt-3 text-xs">
          {copy.setupInstruction}
        </p>
      </AlertDescription>
    </Alert>
  );
}
