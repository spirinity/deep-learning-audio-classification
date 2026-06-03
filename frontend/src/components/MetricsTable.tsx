import type { MetricRow } from "@/lib/types";
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
}

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

export default function MetricsTable({ rows }: Props) {
  if (!rows.length) {
    return (
      <p className="text-sm text-muted-foreground">
        No evaluation results yet. Run{" "}
        <code className="text-primary">python evaluate_methods.py</code> to
        generate <code>comparison_metrics.csv</code>.
      </p>
    );
  }

  return (
    <div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Method</TableHead>
            <TableHead>Accuracy</TableHead>
            <TableHead>Macro F1</TableHead>
            <TableHead>Top-3 Acc</TableHead>
            <TableHead>Avg. inference</TableHead>
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
        Mean across ESC-50 5-fold cross-validation. Inference time is per sample
        on CLAP embeddings (model encoding excluded).
      </p>
    </div>
  );
}
