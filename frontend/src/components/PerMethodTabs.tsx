"use client";

import type { MethodResults, RankItem } from "@/lib/types";
import { categoryColor, INK, methodColor, METHOD_ORDER } from "@/lib/constants";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface Props {
  results: MethodResults;
}

function orderedMethods(results: MethodResults): string[] {
  return [
    ...METHOD_ORDER.filter((m) => m in results),
    ...Object.keys(results).filter(
      (m) => !METHOD_ORDER.includes(m as (typeof METHOD_ORDER)[number])
    ),
  ];
}

function RankBars({ items, color }: { items: RankItem[]; color: string }) {
  return (
    <div className="flex flex-col gap-1.5">
      {items.map((item) => {
        const widthPct = Math.max(1.5, item.score * 100);
        return (
          <div key={item.rank} className="flex items-center gap-3">
            <span className="w-6 shrink-0 text-right text-xs tabular-nums text-muted-foreground">
              {item.rank}
            </span>
            <span className="w-28 shrink-0 truncate text-sm sm:w-36">
              {item.label}
            </span>
            <div className="relative h-5 flex-1 overflow-hidden rounded bg-muted">
              <div
                className="h-full origin-left rounded"
                style={{
                  width: `${widthPct}%`,
                  backgroundColor: color,
                  animation: "barGrow 0.45s ease-out both",
                }}
              />
            </div>
            <span className="w-12 shrink-0 text-right text-xs tabular-nums text-muted-foreground">
              {(item.score * 100).toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

function RankTable({ items }: { items: RankItem[] }) {
  return (
    <Table className="mt-4">
      <TableHeader>
        <TableRow>
          <TableHead>#</TableHead>
          <TableHead>Label</TableHead>
          <TableHead>Score</TableHead>
          <TableHead>Raw</TableHead>
          <TableHead>Category</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {items.map((item) => {
          const color = categoryColor(item.category);
          return (
            <TableRow key={item.rank}>
              <TableCell className="tabular-nums text-muted-foreground">
                {item.rank}
              </TableCell>
              <TableCell className="font-medium">{item.label}</TableCell>
              <TableCell className="tabular-nums">
                {(item.score * 100).toFixed(1)}%
              </TableCell>
              <TableCell className="tabular-nums text-muted-foreground">
                {item.raw_score.toFixed(4)}
              </TableCell>
              <TableCell>
                <span
                  className="inline-flex items-center rounded-full px-2.5 py-0.5 text-[11px] font-semibold tracking-[0.06em] uppercase"
                  style={{ backgroundColor: color, color: INK }}
                >
                  {item.category}
                </span>
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}

export default function PerMethodTabs({ results }: Props) {
  const methods = orderedMethods(results);

  return (
    <Card>
      <CardContent>
        <Tabs defaultValue={methods[0]}>
          <TabsList className="flex flex-wrap">
            {methods.map((method) => (
              <TabsTrigger key={method} value={method}>
                {method}
              </TabsTrigger>
            ))}
          </TabsList>
          {methods.map((method) => (
            <TabsContent key={method} value={method} className="pt-4">
              <RankBars items={results[method]} color={methodColor(method)} />
              <RankTable items={results[method]} />
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
    </Card>
  );
}
