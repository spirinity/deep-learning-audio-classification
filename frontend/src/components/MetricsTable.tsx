import type { MetricRow } from "@/lib/types";
import type { Copy } from "@/lib/i18n";
import { methodColor } from "@/lib/constants";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface Props {
  rows: MetricRow[];
  copy: Copy;
}

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

export default function MetricsTable({ rows, copy }: Props) {
  if (!rows.length) {
    return (
      <p className="text-sm text-muted-foreground">
        {copy.noMetrics}
      </p>
    );
  }

  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>{copy.metricMethod}</TableHead>
            <TableHead>{copy.metricAccuracy}</TableHead>
            <TableHead>{copy.metricMacroF1}</TableHead>
            <TableHead>{copy.metricTop3}</TableHead>
            <TableHead>{copy.metricInference}</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((r) => (
            <TableRow key={r.method}>
              <TableCell className="font-medium">
                <span className="flex items-center gap-2">
                  <span
                    className="size-2.5 rounded-full"
                    style={{ backgroundColor: methodColor(r.method) }}
                  />
                  {r.method}
                </span>
              </TableCell>
              <TableCell className="tabular-nums">{pct(r.accuracy)}</TableCell>
              <TableCell className="tabular-nums text-muted-foreground">
                {pct(r.macro_f1)}
              </TableCell>
              <TableCell className="tabular-nums text-muted-foreground">
                {pct(r.top3_accuracy)}
              </TableCell>
              <TableCell className="tabular-nums text-muted-foreground">
                {(r.avg_inference_time_sec * 1000).toFixed(2)} ms
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <p className="mt-3 text-xs text-muted-foreground">
        {copy.metricsFootnote}
      </p>
    </div>
  );
}
