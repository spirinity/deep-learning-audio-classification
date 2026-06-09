"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { Copy, Language } from "@/lib/i18n";
import { LanguagesIcon } from "lucide-react";

interface Props {
  copy: Copy;
  language: Language;
  onLanguageChange: (language: Language) => void;
}

export default function Header({ copy, language, onLanguageChange }: Props) {
  return (
    <header className="flex flex-col items-center text-center">
      <div className="mb-5 flex w-full items-center justify-center">
        <div
          className="inline-flex items-center gap-1 rounded-lg border border-border bg-card p-1"
          aria-label={copy.language}
        >
          <LanguagesIcon className="mx-2 size-4 text-muted-foreground" />
          {(["id", "en"] as const).map((lang) => (
            <Button
              key={lang}
              type="button"
              size="sm"
              variant={language === lang ? "default" : "ghost"}
              onClick={() => onLanguageChange(lang)}
              className="h-8 px-3 text-xs font-semibold"
              aria-pressed={language === lang}
            >
              {lang.toUpperCase()}
            </Button>
          ))}
        </div>
      </div>

      <Badge
        variant="secondary"
        className="mb-5 text-[11px] font-semibold tracking-[0.08em] text-muted-foreground uppercase"
      >
        INTERSPEECH 2023 - LAION-CLAP - ESC-50
      </Badge>
      <h1 className="display text-4xl text-foreground sm:text-6xl">
        {copy.headerTitle}
      </h1>
      <p className="mt-4 max-w-2xl text-base text-muted-foreground sm:text-lg">
        {copy.headerPrefix}{" "}
        <span className="text-foreground">Zero-Shot CLAP</span>,{" "}
        <span className="text-foreground">Proto-LC</span>, {copy.headerMiddle}{" "}
        <span className="text-foreground">Logistic Regression</span>{" "}
        {copy.headerSuffix}
      </p>
    </header>
  );
}
