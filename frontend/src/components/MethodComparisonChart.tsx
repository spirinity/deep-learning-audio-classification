import type { MethodResults } from "@/lib/types";
import { methodColor, METHOD_ORDER } from "@/lib/constants";
import { Card, CardContent } from "@/components/ui/card";

interface Props {
  results: MethodResults;
}

// Grouped top-1 confidence per method, drawn as pure-CSS vertical bars.
// Solid pastel fills (no gradients) per the Cursor design language.
export default function MethodComparisonChart({ results }: Props) {
  const methods = [
    ...METHOD_ORDER.filter((m) => m in results),
    ...Object.keys(results).filter(
      (m) => !METHOD_ORDER.includes(m as (typeof METHOD_ORDER)[number])
    ),
  ];

  return (
    <Card>
      <CardContent className="py-2">
        <div className="flex items-end justify-around gap-4 border-b border-border pb-3" style={{ height: 200 }}>
          {methods.map((method) => {
            const top = results[method]?.[0];
            if (!top) return null;
            const color = methodColor(method);
            const heightPct = Math.max(2, top.score * 100);
            return (
              <div
                key={method}
                className="flex h-full flex-1 flex-col items-center justify-end gap-2"
              >
                <span className="font-mono text-sm text-foreground">
                  {(top.score * 100).toFixed(1)}%
                </span>
                <div
                  className="w-full max-w-[72px] origin-bottom rounded-t-md"
                  style={{
                    height: `${heightPct}%`,
                    backgroundColor: color,
                    animation: "barRise 0.5s ease-out both",
                  }}
                  title={`${method}: ${top.label} (${(top.score * 100).toFixed(1)}%)`}
                />
              </div>
            );
          })}
        </div>
        <div className="flex justify-around gap-4 pt-3">
          {methods.map((method) => {
            const top = results[method]?.[0];
            if (!top) return null;
            return (
              <div key={method} className="flex flex-1 flex-col items-center gap-0.5 text-center">
                <span className="max-w-[120px] truncate text-sm text-foreground">
                  {top.label}
                </span>
                <span className="text-[10px] tracking-[0.06em] text-muted-foreground uppercase">
                  {method}
                </span>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
