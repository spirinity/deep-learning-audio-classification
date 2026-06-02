import type { MethodResults } from "@/lib/types";
import { categoryColor, INK, METHOD_ORDER } from "@/lib/constants";
import { Card, CardContent } from "@/components/ui/card";

interface Props {
  results: MethodResults;
}

function orderedMethods(results: MethodResults): string[] {
  const known = METHOD_ORDER.filter((m) => m in results);
  const extra = Object.keys(results).filter(
    (m) => !METHOD_ORDER.includes(m as (typeof METHOD_ORDER)[number])
  );
  return [...known, ...extra];
}

export default function ResultCards({ results }: Props) {
  const methods = orderedMethods(results);

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      {methods.map((method) => {
        const top = results[method]?.[0];
        if (!top) return null;
        const color = categoryColor(top.category);
        return (
          <Card key={method} className="animate-fade-in-up">
            <CardContent className="flex min-h-[168px] flex-col items-center justify-center gap-2 px-4 py-6 text-center">
              <span className="text-[11px] font-semibold tracking-[0.08em] text-muted-foreground uppercase">
                {method}
              </span>
              <span className="display text-2xl text-foreground">
                {top.label}
              </span>
              <span className="font-mono text-lg text-foreground">
                {(top.score * 100).toFixed(1)}%
              </span>
              <span
                className="mt-1 inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-semibold tracking-[0.06em] uppercase"
                style={{ backgroundColor: color, color: INK }}
              >
                {top.category}
              </span>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
